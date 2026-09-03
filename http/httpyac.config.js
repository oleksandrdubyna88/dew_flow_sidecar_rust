// Variables for the `.http` contract suite.
//
// The sidecar is unauthenticated by design — it binds 127.0.0.1 and is spoken to by one host on the
// same machine — so there is nothing to mint here. The only thing worth configuring is where it is.

module.exports = {
  environments: {
    local: {
      baseUrl: process.env.SIDECAR_BASE_URL ?? 'http://127.0.0.1:5320',
    },
  },
};
