# treefmt.nix
{ pkgs, ... }:
{
  # Used to find the project root
  projectRootFile = "flake.nix";
  programs = builtins.mapAttrs (k: v: v // { enable = true; }) {
    black = { };
    nixfmt = { };
  };
}
