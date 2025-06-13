{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };
  outputs =
    {
      flake-utils,
      nixpkgs,
      self,
      treefmt-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        treefmt = treefmt-nix.lib.evalModule pkgs ./.treefmt.nix;

      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            let
              python = pkgs.python3.withPackages (
                p: with p; [
                  beartype
                  brax
                  flax
                  jax-cuda12-plugin
                  jax-cuda12-pjrt
                  jaxtyping
                  opencv-python
                  mediapy
                  mujoco-mjx
                ]
              );
            in
            [
              python
            ]
            ++ (with pkgs; [
              black
              ffmpeg
            ]);
        };

        formatter = treefmt.config.build.wrapper;

        MUJOCO_GL = "egl";
      }
    );
}
