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
            # let
            #   # These packages get built by Nix, and will be ahead on the PATH
            #   pwp = (
            #     python312.withPackages (
            #       p: with p; [
            #         venvShellHook
            #       ]
            #     )
            #   );
            # in
            mkShell {
              NIX_LD = runCommand "ld.so" { } ''
                ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
              '';
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                python312Packages.venvShellHook
                uv
                pkgs.gcc
                claude-code
                # python312Packages.venvShellHook
		duckdb
              ]
              ++ libList;
              shellHook = ''
                # if [ ! -d .venv ]; then
                #   source .venv/bin/activate
                # fi
                export PYTHONPATH=.venv/lib/python3.12/site-packages/
                [ -f /etc/os-release ] && grep -q "ID=nixos" /etc/os-release && echo export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:"/run/opengl-driver/lib":$LD_LIBRARY_PATH
                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

                uv sync --all-groups
                source .venv/bin/activate
                # runHook venvShellHook
              '';
            };
        };
      }
    );
}
                # # Set up CUDA environment variables
                # export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}

                # # Find NVIDIA driver libraries
                # NVIDIA_DRIVER_LIB=$(find /nix/store -name "libcuda.so.1" 2>/dev/null | head -1 | xargs dirname)
                # export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$NVIDIA_DRIVER_LIB:$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                # export EXTRA_LDFLAGS="-L${pkgs.cudaPackages.cudatoolkit}/lib"
                # export EXTRA_CCFLAGS="-I${pkgs.cudaPackages.cudatoolkit}/include"

                # Create and activate venv
                # if [ ! -d .venv ]; then
                  # ${pkgs.python312}/bin/python -m venv .venv
                # fi

                # Install PyTorch with CUDA support
                # uv pip install torch torchvision torchmetrics --index-url https://download.pytorch.org/whl/cu121

                # Add PyTorch lib path
                # export LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
