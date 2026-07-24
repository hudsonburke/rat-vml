{
  description = "VML gait analysis and AGU manuscript";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          tools = with pkgs; [
            pyright
            ruff
            uv
            git
            cmake
            gcc
            gnumake
            quarto
          ];
        in
        {
          default = pkgs.mkShell {
            packages = tools;
            shellHook = ''
              echo "rat-vml dev shell"
              echo "  uv sync              — install core deps"
              echo "  uv sync --extra ingest — install C3D ingestion deps"
              echo "  python scripts/ingest.py pull   — download .rrd catalog from HF"
              echo "  python scripts/catalog.py query — query valid walking trials"
              echo "  python scripts/run_analysis.py  — run full analysis pipeline"
              echo "  quarto render         — build the AGU manuscript"
            '';
          };
        });
    };
}
