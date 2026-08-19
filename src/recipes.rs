//! The build recipe, kept honest against the crate it describes.
//!
//! `build-recipes.json` is the machine-readable answer to "what do I install and what do I run" — read by
//! the RAG side's compile button and by a human, which is why it is checked in rather than served: it is
//! needed BEFORE any sidecar exists.
//!
//! A table like that decays the moment a feature is renamed, and it decays SILENTLY, because nothing
//! compiles it. So the file is parsed here and checked against `Cargo.toml`: every feature a recipe names
//! must exist, every flavour this binary can be built as must have a row, and no row may claim to ship
//! something the licence position forbids shipping. Those are the three ways it could start lying.
//!
//! There is no runtime code in this module. It exists to be tested.

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    const RECIPES: &str = include_str!("../build-recipes.json");
    const MANIFEST: &str = include_str!("../Cargo.toml");

    #[derive(Deserialize)]
    struct Recipes {
        schema: String,
        recipes: Vec<Recipe>,
    }

    #[derive(Deserialize)]
    struct Recipe {
        id: String,
        title: String,
        platform: String,
        command: String,
        cargo_features: Vec<String>,
        toolchain: Vec<String>,
        is_default: bool,
        ships_freely: bool,
        why_not_shipped: Option<String>,
        build_minutes: Option<f64>,
        verify: String,
    }

    fn recipes() -> Recipes {
        serde_json::from_str(RECIPES).expect("build-recipes.json parses")
    }

    /// Every flavour this crate can actually be built as needs a row. A provider a customer's card wants and
    /// the recipe does not mention is a provider they cannot be told how to build.
    #[test]
    fn every_flavour_the_crate_offers_has_a_recipe() {
        let all = recipes();
        let ids: Vec<&str> = all.recipes.iter().map(|r| r.id.as_str()).collect();

        for expected in ["windows-directml", "windows-cuda", "linux-migraphx", "cpu"] {
            assert!(
                ids.contains(&expected),
                "no recipe for `{expected}`; rows are {ids:?}"
            );
        }
        assert_eq!(
            ids.len(),
            ids.iter().collect::<std::collections::HashSet<_>>().len(),
            "two rows share an id, so a caller picking by id gets whichever it finds first"
        );
    }

    /// THE decay this file exists to prevent. A recipe naming a cargo feature the crate does not have sends
    /// an operator to a command that fails, and nothing else in the build would notice — the JSON compiles
    /// as prose.
    #[test]
    fn every_feature_a_recipe_names_exists_in_the_manifest() {
        let declared = manifest_features();

        for recipe in recipes().recipes {
            for feature in &recipe.cargo_features {
                assert!(
                    declared.contains(feature),
                    "recipe `{}` names cargo feature `{feature}`, which Cargo.toml does not declare \
                     (declared: {declared:?})",
                    recipe.id
                );
            }
        }
    }

    /// And the reverse, which is the half that rots quietly: a feature added to the crate with no recipe is
    /// a flavour nobody outside this repository knows how to build.
    #[test]
    fn every_feature_the_manifest_declares_appears_in_some_recipe() {
        let named: Vec<String> = recipes()
            .recipes
            .iter()
            .flat_map(|r| r.cargo_features.clone())
            .collect();

        for feature in manifest_features() {
            // `default` is the DirectML row, which names no features because it needs none.
            if feature == "default" || feature == "dml" {
                continue;
            }
            assert!(
                named.contains(&feature),
                "Cargo.toml declares feature `{feature}` and no recipe builds with it"
            );
        }
    }

    /// The command a recipe prints must match the features it declares, or the two halves of the row
    /// disagree and the operator follows the one they can copy.
    #[test]
    fn the_command_and_the_declared_features_say_the_same_thing() {
        for recipe in recipes().recipes {
            for feature in &recipe.cargo_features {
                assert!(
                    recipe.command.contains(&format!("--features {feature}"))
                        || recipe.command.contains(&format!("--features \"{feature}")),
                    "recipe `{}` declares feature `{feature}` and its command does not pass it: {}",
                    recipe.id,
                    recipe.command
                );
            }

            if recipe.cargo_features.is_empty() && !recipe.is_default {
                assert!(
                    recipe.command.contains("--no-default-features"),
                    "recipe `{}` names no features and is not the default, so its command must turn the \
                     default off: {}",
                    recipe.id,
                    recipe.command
                );
            }
        }
    }

    /// Exactly one default. Two would make "what do I build here" ambiguous; none would make it unanswerable.
    #[test]
    fn exactly_one_recipe_is_the_default() {
        let defaults: Vec<String> = recipes()
            .recipes
            .into_iter()
            .filter(|r| r.is_default)
            .map(|r| r.id)
            .collect();

        assert_eq!(defaults, vec!["windows-directml".to_string()]);
    }

    /// The licence position, asserted where it can be broken by an edit. Only the CPU flavour carries no
    /// vendor component, so only it may ship — and anything that may NOT ship has to say why, because
    /// "no" without a reason is the kind of rule someone quietly relaxes.
    #[test]
    fn only_the_cpu_flavour_ships_and_every_other_row_says_why_not() {
        for recipe in recipes().recipes {
            if recipe.id == "cpu" {
                assert!(
                    recipe.ships_freely,
                    "the CPU build is the one that carries no vendor component"
                );
                continue;
            }

            assert!(
                !recipe.ships_freely,
                "recipe `{}` claims it ships freely",
                recipe.id
            );
            assert!(
                recipe
                    .why_not_shipped
                    .as_deref()
                    .is_some_and(|why| why.len() > 30),
                "recipe `{}` does not ship and does not say why",
                recipe.id
            );
        }
    }

    /// Every row has to be readable by a human as well as a machine: a title, a platform, a toolchain list
    /// and a way to tell whether the result works. An empty field here is a row that answers nothing.
    #[test]
    fn every_row_is_answerable_by_a_human() {
        for recipe in recipes().recipes {
            assert!(!recipe.title.is_empty(), "{} has no title", recipe.id);
            assert!(!recipe.platform.is_empty(), "{} has no platform", recipe.id);
            assert!(!recipe.command.is_empty(), "{} has no command", recipe.id);
            assert!(
                !recipe.toolchain.is_empty(),
                "{} lists no toolchain",
                recipe.id
            );
            assert!(
                recipe.verify.contains("/health"),
                "{} does not say how to tell whether the build works",
                recipe.id
            );
        }
    }

    /// A build time is MEASURED or it is absent. An invented number is worse than none: it is the field an
    /// operator uses to decide whether to start a build now or after lunch.
    #[test]
    fn a_build_time_is_a_measurement_or_it_is_absent() {
        for recipe in recipes().recipes {
            if let Some(minutes) = recipe.build_minutes {
                assert!(
                    minutes > 0.0 && minutes < 600.0,
                    "recipe `{}` reports {minutes} minutes, which is not a plausible measurement",
                    recipe.id
                );
            }
        }
    }

    #[test]
    fn the_schema_is_versioned_so_a_consumer_can_refuse_a_shape_it_does_not_know() {
        assert_eq!(recipes().schema, "dewflow.sidecar.build-recipes/v1");
    }

    /// The `[features]` table of Cargo.toml, read as text rather than parsed: this crate has no TOML
    /// dependency and adding one to read seven lines would be a package taken for a test.
    fn manifest_features() -> Vec<String> {
        MANIFEST
            .lines()
            .skip_while(|line| line.trim() != "[features]")
            .skip(1)
            .take_while(|line| !line.trim_start().starts_with('['))
            .filter_map(|line| {
                line.split_once('=')
                    .map(|(name, _)| name.trim().to_string())
            })
            .filter(|name| !name.is_empty() && !name.starts_with('#'))
            .collect()
    }

    /// The reader above is itself a guess about Cargo.toml's shape, so it is checked: if it ever returns
    /// nothing, every assertion built on it passes vacuously.
    #[test]
    fn the_manifest_reader_actually_finds_the_features() {
        let features = manifest_features();

        assert!(
            features.contains(&"default".to_string()),
            "found {features:?}"
        );
        assert!(features.contains(&"cuda".to_string()));
        assert!(features.contains(&"migraphx".to_string()));
    }
}
