{
  description = "Local TTS API with a configurable NVIDIA-ready NixOS service";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      chorus = pkgs.callPackage ./nix/package.nix { };
      development = (import ./nix/flake.nix).outputs { inherit self nixpkgs; };
    in
    {
      inherit (development) devShells;

      packages.${system} = {
        default = chorus;
        inherit chorus;
      };

      nixosModules = {
        default = import ./nix/module.nix;
        chorus = self.nixosModules.default;
      };
    };
}
