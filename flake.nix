{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          system = system;
          config.allowUnfree = true;
        };

        libList = [
          # Add needed packages here
          pkgs.libz # Numpy
          pkgs.stdenv.cc.cc
          pkgs.libGL
          pkgs.glib
          # CUDA packages
          pkgs.cudaPackages.cudatoolkit
          pkgs.cudaPackages.cudnn
        ];
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              # These packages get built by Nix, and will be ahead on the PATH
              pwp = (
                python312.withPackages (
                  p: with p; [
                    venvShellHook
                  ]
                )
              );
            in
            mkShell {
              NIX_LD = runCommand "ld.so" { } ''
                ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
              '';
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                pwp
                uv
                pkgs.gcc
                claude-code
              ]
              ++ libList;
              shellHook = ''
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:"/run/opengl-driver/lib":$LD_LIBRARY_PATH

                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

                # Set up CUDA environment variables
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}

                # Cache the NVIDIA driver path
                NVIDIA_CACHE=".nix-cache/nvidia-driver-path"
                if [ ! -f "$NVIDIA_CACHE" ]; then
                  mkdir -p .nix-cache
                  find /nix/store -name "libcuda.so.1" 2>/dev/null | head -1 | xargs dirname > "$NVIDIA_CACHE"
                fi
                NVIDIA_DRIVER_LIB=$(cat "$NVIDIA_CACHE")

                export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$NVIDIA_DRIVER_LIB:$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                export EXTRA_LDFLAGS="-L${pkgs.cudaPackages.cudatoolkit}/lib"
                export EXTRA_CCFLAGS="-I${pkgs.cudaPackages.cudatoolkit}/include"

                # Create venv if it doesn't exist
                if [ ! -d .venv ]; then
                  ${pkgs.python312}/bin/python3.12 -m venv .venv
                  source .venv/bin/activate
                  uv sync --all-groups
                  uv pip install torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cu121
                  # Mark that we've installed
                  touch .nix-cache/deps-installed
                else
                  source .venv/bin/activate
                  # Only re-sync if pyproject.toml or uv.lock changed since last install
                  if [ pyproject.toml -nt .nix-cache/deps-installed ] || [ uv.lock -nt .nix-cache/deps-installed ]; then
                    echo "Dependencies changed, re-syncing..."
                    uv sync --all-groups
                    touch .nix-cache/deps-installed
                  fi
                fi

                # Add PyTorch lib path
                export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
              '';
            };
        };
      }
    );
}
