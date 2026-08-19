//! DXGI adapter resolution for the DirectML execution provider.
//!
//! Two different numbering spaces exist for the same physical adapters:
//!   * The OPERATOR-FACING device id (`ORT_DEVICE_ID`, the UI picker's "GPU {id}") counts hardware
//!     adapters in DXGI *high-performance* order (`IDXGIFactory6::EnumAdapterByGpuPreference`),
//!     the same ordering ONNX Runtime's EP-device discovery reports as `DxgiHighPerformanceIndex`
//!     and the host's DirectML lister uses for the picker labels. Device 0 = the fastest card.
//!   * The LEGACY DirectML EP (`SessionOptionsAppendExecutionProvider_DML`, what
//!     `ort`'s `with_device_id` calls) indexes adapters in *plain* `EnumAdapters1` order — on a
//!     machine whose display runs on an integrated GPU that is typically the iGPU, not the
//!     discrete card. Feeding the operator id straight into it silently ran inference on the
//!     wrong adapter while the picker claimed another.
//!
//! `resolve` translates: high-performance index -> adapter LUID -> plain-enumeration index, and
//! reports the adapter it lands on so /health can show the ground truth.

#[derive(Clone, Debug, serde::Serialize)]
pub struct ResolvedAdapter {
    /// DXGI adapter description, e.g. "AMD Radeon AI PRO R9700".
    pub name: String,
    pub vram_mb: u64,
    /// Adapter LUID as hex — stable within a boot, unique per adapter.
    pub luid: String,
    /// The ORT_DEVICE_ID this sidecar serves (high-performance order — the picker's numbering).
    pub requested_device: i32,
    /// The index handed to the DirectML EP (plain EnumAdapters order — its legacy numbering).
    pub dml_device_id: i32,
}

/// One adapter as either enumeration lists it. LUID is the identity that links the two orders.
/// Consumed by the Windows-only DXGI path and by the (OS-independent) mapping tests — dead code
/// only in a non-Windows, non-test build.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdapterDesc {
    pub luid: i64,
    pub name: String,
    pub vram_mb: u64,
    pub software: bool,
}

/// Maps an operator device id (index into the high-performance-ordered HARDWARE adapters) to the
/// plain-enumeration index the legacy DirectML EP consumes. Pure so the ordering contract is
/// testable without DXGI. The plain list stays unfiltered: the EP counts software entries too.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn map_device(
    hp: &[AdapterDesc],
    plain: &[AdapterDesc],
    device: usize,
) -> Option<(usize, AdapterDesc)> {
    let target = hp.iter().filter(|a| !a.software).nth(device)?;
    let dml_index = plain.iter().position(|p| p.luid == target.luid)?;
    Some((dml_index, target.clone()))
}

/// Resolves the sidecar's `ORT_DEVICE_ID` against the machine's DXGI adapters. `None` means the
/// mapping could not be established (non-Windows build, DXGI failure, id past the adapter count) —
/// the caller falls back to passing the raw id through, which is the pre-mapping behaviour.
#[cfg(windows)]
pub fn resolve(device_id: i32) -> Option<ResolvedAdapter> {
    if device_id < 0 {
        return None;
    }
    let hp = dxgi::enumerate_high_performance()?;
    let plain = dxgi::enumerate_plain()?;
    let (dml_index, adapter) = map_device(&hp, &plain, device_id as usize)?;
    Some(ResolvedAdapter {
        name: adapter.name,
        vram_mb: adapter.vram_mb,
        luid: format!("{:#x}", adapter.luid),
        requested_device: device_id,
        dml_device_id: dml_index as i32,
    })
}

#[cfg(not(windows))]
pub fn resolve(_device_id: i32) -> Option<ResolvedAdapter> {
    None // DirectML is Windows-only; other builds (CUDA/CPU) keep the raw id semantics.
}

/// How many bytes THIS PROCESS currently holds in the adapter's local memory segment, or `None` when
/// there is no way to ask.
///
/// `plain_index` is the DirectML EP's own numbering — the index `ResolvedAdapter::dml_device_id`
/// carries — so the sample is taken against the card the engines actually run on rather than against
/// whichever adapter enumerates first. Pass the resolved index; a caller that has no resolved adapter
/// has no business guessing one, and `None` is the honest answer there.
///
/// `None` is never a zero. A process that holds no VRAM and a process that could not be asked are
/// different facts, and the field this feeds exists precisely to keep them apart.
#[cfg(windows)]
pub(crate) fn process_vram_bytes(plain_index: i32) -> Option<u64> {
    dxgi::process_local_usage(u32::try_from(plain_index).ok()?)
}

/// DXGI is Windows. The MIGraphX (WSL/ROCm) and CUDA-on-Linux flavours have no equivalent here, and
/// that is a stated limit rather than a gap to fill later with a second, differently-scaled mechanism.
#[cfg(not(windows))]
pub(crate) fn process_vram_bytes(_plain_index: i32) -> Option<u64> {
    None
}

#[cfg(windows)]
mod dxgi {
    use super::AdapterDesc;
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory6,
        DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
    };

    /// Hardware-and-software adapters in "fastest first" order (the picker's numbering source).
    pub fn enumerate_high_performance() -> Option<Vec<AdapterDesc>> {
        let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory1() }.ok()?;
        enumerate("EnumAdapterByGpuPreference", |index| unsafe {
            factory.EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
        })
    }

    /// The same adapters in plain enumeration order — the numbering the legacy DirectML EP indexes.
    pub fn enumerate_plain() -> Option<Vec<AdapterDesc>> {
        let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory1() }.ok()?;
        enumerate("EnumAdapters1", |index| unsafe {
            factory.EnumAdapters1(index)
        })
    }

    /// Shared walk: indices until DXGI_ERROR_NOT_FOUND ends the list; any other failure (or an
    /// undescribable adapter) yields None so the caller falls back to the raw device id.
    fn enumerate(
        method: &str,
        get: impl Fn(u32) -> windows::core::Result<IDXGIAdapter1>,
    ) -> Option<Vec<AdapterDesc>> {
        let mut adapters = Vec::new();
        for index in 0..64u32 {
            let adapter = match get(index) {
                Ok(adapter) => adapter,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => {
                    tracing::warn!("DXGI {method}({index}) failed: {e}");
                    return None;
                }
            };
            adapters.push(describe(&adapter)?);
        }
        Some(adapters)
    }

    /// `IDXGIAdapter3::QueryVideoMemoryInfo` for the LOCAL segment — the one figure DXGI reports about
    /// THIS process rather than about the machine, which is what makes a before/after delta around a
    /// build attributable at all. `IDXGIAdapter3` is a cast of the adapter we already enumerate; a
    /// driver too old to serve it yields `None` like every other failure here.
    pub fn process_local_usage(index: u32) -> Option<u64> {
        let factory: IDXGIFactory6 = unsafe { CreateDXGIFactory1() }.ok()?;
        let adapter: IDXGIAdapter3 = unsafe { factory.EnumAdapters1(index) }.ok()?.cast().ok()?;
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .ok()?;
        Some(info.CurrentUsage)
    }

    fn describe(adapter: &IDXGIAdapter1) -> Option<AdapterDesc> {
        let desc = unsafe { adapter.GetDesc1() }.ok()?;
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        Some(AdapterDesc {
            luid: ((desc.AdapterLuid.HighPart as i64) << 32) | (desc.AdapterLuid.LowPart as i64),
            name: String::from_utf16_lossy(&desc.Description[..len]),
            vram_mb: (desc.DedicatedVideoMemory / (1024 * 1024)) as u64,
            software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{map_device, AdapterDesc};

    fn adapter(luid: i64, name: &str, software: bool) -> AdapterDesc {
        AdapterDesc {
            luid,
            name: name.into(),
            vram_mb: 0,
            software,
        }
    }

    /// The reported defect: plain order lists the iGPU first (it drives the display), the
    /// performance order lists the discrete card first. Device 0 must land on the discrete card's
    /// PLAIN index, not on plain index 0.
    #[test]
    fn device_zero_maps_to_the_fastest_card_even_when_it_enumerates_second() {
        let hp = [
            adapter(2, "Discrete", false),
            adapter(1, "Integrated", false),
        ];
        let plain = [
            adapter(1, "Integrated", false),
            adapter(2, "Discrete", false),
        ];

        let (dml, chosen) = map_device(&hp, &plain, 0).expect("device 0 must resolve");
        assert_eq!(dml, 1);
        assert_eq!(chosen.name, "Discrete");

        let (dml, chosen) = map_device(&hp, &plain, 1).expect("device 1 must resolve");
        assert_eq!(dml, 0);
        assert_eq!(chosen.name, "Integrated");
    }

    /// Software adapters (Microsoft Basic Render Driver) never take a device slot in the operator
    /// numbering, but they DO occupy an index in the plain order the DirectML EP counts.
    #[test]
    fn software_adapter_is_skipped_in_device_numbering_but_counted_in_plain_order() {
        let hp = [
            adapter(9, "Basic Render", true),
            adapter(2, "Discrete", false),
        ];
        let plain = [
            adapter(9, "Basic Render", true),
            adapter(2, "Discrete", false),
        ];

        let (dml, chosen) = map_device(&hp, &plain, 0).expect("device 0 must resolve");
        assert_eq!(dml, 1);
        assert_eq!(chosen.name, "Discrete");
    }

    #[test]
    fn device_id_past_the_hardware_adapter_count_does_not_resolve() {
        let hp = [adapter(2, "Discrete", false)];
        let plain = [adapter(2, "Discrete", false)];
        assert!(map_device(&hp, &plain, 1).is_none());
    }

    #[test]
    fn an_adapter_missing_from_the_plain_order_does_not_resolve() {
        let hp = [adapter(2, "Discrete", false)];
        let plain = [adapter(1, "Integrated", false)];
        assert!(map_device(&hp, &plain, 0).is_none());
    }
}
