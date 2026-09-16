{
  config,
  lib,
  pkgs,
  utils,
  ...
}:

let
  cfg = config.services.chorus;

  supportedEngines = [
    "kokoro"
    "breeze"
    "fish"
    "piper"
    "kitten"
    "pocket"
    "supertonic"
  ];

  engineType = lib.types.enum supportedEngines;

  # Paths are also used in tmpfiles rules, which have their own quoting syntax.
  validDirectory =
    path:
    lib.hasPrefix "/" path
    && lib.all (part: part != "." && part != ".." && builtins.match "[A-Za-z0-9._-]+" part != null) (
      lib.tail (lib.splitString "/" path)
    );

  deviceIsValid = device: device == "cpu" || builtins.match "cuda:[0-9]+" device != null;

  vramArgs = lib.concatLists (
    lib.mapAttrsToList (device: mib: [
      "--vram-budget-mib"
      "${device}=${toString mib}"
    ]) cfg.vramBudgetMiB
  );

  cliArgs = [
    "${cfg.package}/bin/chorus-service"
    "--host"
    cfg.host
    "--port"
    (toString cfg.port)
    "--devices"
    (lib.concatStringsSep "," cfg.devices)
    "--idle-timeout"
    (toString cfg.idleTimeout)
    "--gpu-queue-size"
    (toString cfg.gpuQueueSize)
  ]
  ++ lib.optional (cfg.ramBudgetMiB != null) "--ram-budget-mib"
  ++ lib.optional (cfg.ramBudgetMiB != null) (toString cfg.ramBudgetMiB)
  ++ vramArgs
  ++ lib.concatMap (selector: [
    "--preload"
    selector
  ]) cfg.preload
  ++ lib.optional (cfg.channelConfig != null) "--channel-config"
  ++ lib.optional (cfg.channelConfig != null) cfg.channelConfig
  ++ lib.optional cfg.downloadMissing "--download-missing"
  ++ cfg.extraArgs;

  protectedEnvironment = {
    CHORUS_STATE_DIR = cfg.stateDirectory;
    CHORUS_CACHE_DIR = cfg.cacheDirectory;
    CHORUS_ENGINES = lib.concatStringsSep "," cfg.engines;
    CHORUS_MODELS_DIR = cfg.modelsDirectory;
    # Keep transient files and model/download caches inside the explicitly
    # writable service paths.  PrivateTmp additionally isolates /tmp.
    TMPDIR = "/run/chorus";
    XDG_CACHE_HOME = cfg.cacheDirectory;
    UV_CACHE_DIR = "${cfg.cacheDirectory}/uv";
    HF_HOME = "${cfg.cacheDirectory}/huggingface";
  };
in
{
  options.services.chorus = {
    enable = lib.mkEnableOption "the Chorus multi-engine TTS API service";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { };
      defaultText = lib.literalExpression "pkgs.callPackage ./package.nix { }";
      description = "Package providing the chorus-service launcher.";
    };

    engines = lib.mkOption {
      type = lib.types.listOf engineType;
      default = [ "kokoro" ];
      description = ''
        Engines enabled by the service.  The launcher receives this selection
        through CHORUS_ENGINES.  Supported values are kokoro, breeze, fish,
        piper, kitten, pocket, and supertonic.
      '';
    };

    devices = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [
        "cpu"
        "cuda:0"
      ];
      description = "Inference devices, such as cpu and cuda:0.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address on which Chorus listens (loopback by default).";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "TCP port on which Chorus listens.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the configured TCP port in the system firewall.";
    };

    downloadMissing = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Download missing selected-engine model files during service startup.
        Downloads are performed at runtime, never while evaluating or building
        the Nix derivation.
      '';
    };

    stateDirectory = lib.mkOption {
      type = lib.types.str;
      default = "/var/lib/chorus";
      description = ''
        Writable persistent state directory.  The default is the canonical
        systemd state path; model manifests and the mutable application live
        below this directory.
      '';
    };

    cacheDirectory = lib.mkOption {
      type = lib.types.str;
      default = "/var/cache/chorus";
      description = ''
        Writable persistent cache directory.  The default is the canonical
        systemd cache path; all transient/download caches are confined here.
      '';
    };
    modelsDirectory = lib.mkOption {
      type = lib.types.str;
      default = "${cfg.stateDirectory}/models";
      defaultText = lib.literalExpression "config.services.chorus.stateDirectory + \"/models\"";
      description = "Writable model catalog and weights directory (defaults to stateDirectory/models).";
    };

    idleTimeout = lib.mkOption {
      type = lib.types.number;
      default = 300.0;
      description = "Seconds after which an idle model worker may be unloaded.";
    };

    gpuQueueSize = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 32;
      description = "Maximum number of GPU requests waiting for a worker slot.";
    };

    ramBudgetMiB = lib.mkOption {
      type = lib.types.nullOr lib.types.number;
      default = null;
      description = "Optional aggregate model RAM cache budget in MiB.";
    };

    vramBudgetMiB = lib.mkOption {
      type = lib.types.attrsOf lib.types.number;
      default = { };
      example = {
        "cuda:0" = 4096.0;
      };
      description = "Optional per-CUDA-device model VRAM cache budgets in MiB.";
    };

    preload = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "kokoro/82m-v1.0" ];
      description = "Models to preload, using ENGINE/MODEL selectors.";
    };

    channelConfig = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional path to a TOML per-model channel policy configuration.";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Additional chorus-service command-line arguments, passed verbatim.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Additional environment variables for the Chorus service.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional systemd EnvironmentFile containing secrets and other runtime settings.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = pkgs.stdenv.hostPlatform.system == "x86_64-linux";
        message = "services.chorus supports x86_64-linux only.";
      }
      {
        assertion = cfg.engines != [ ];
        message = "services.chorus.engines must not be empty.";
      }
      {
        assertion = lib.length cfg.engines == lib.length (lib.unique cfg.engines);
        message = "services.chorus.engines must not contain duplicates.";
      }
      {
        assertion = cfg.devices != [ ];
        message = "services.chorus.devices must not be empty.";
      }
      {
        assertion = lib.length cfg.devices == lib.length (lib.unique cfg.devices);
        message = "services.chorus.devices must not contain duplicates.";
      }
      {
        assertion = lib.all deviceIsValid cfg.devices;
        message = "services.chorus.devices must contain only cpu or cuda:N device IDs.";
      }
      {
        assertion = validDirectory cfg.stateDirectory;
        message = "services.chorus.stateDirectory must be a normalized absolute path using letters, digits, dots, underscores, and hyphens.";
      }
      {
        assertion = validDirectory cfg.cacheDirectory;
        message = "services.chorus.cacheDirectory must be a normalized absolute path using letters, digits, dots, underscores, and hyphens.";
      }
      {
        assertion = validDirectory cfg.modelsDirectory;
        message = "services.chorus.modelsDirectory must be a normalized absolute path using letters, digits, dots, underscores, and hyphens.";
      }
      {
        assertion = cfg.idleTimeout >= 0.0;
        message = "services.chorus.idleTimeout must be nonnegative.";
      }
      {
        assertion = cfg.ramBudgetMiB == null || cfg.ramBudgetMiB > 0.0;
        message = "services.chorus.ramBudgetMiB must be positive when set.";
      }
      {
        assertion = lib.all (mib: mib > 0.0) (lib.attrValues cfg.vramBudgetMiB);
        message = "services.chorus.vramBudgetMiB values must be positive.";
      }
      {
        assertion = lib.all (device: lib.hasPrefix "cuda:" device && lib.elem device cfg.devices) (
          lib.attrNames cfg.vramBudgetMiB
        );
        message = "services.chorus.vramBudgetMiB keys must be enabled cuda:N devices.";
      }
      {
        assertion = lib.all (selector: builtins.match "[^/]+/[^/]+" selector != null) cfg.preload;
        message = "services.chorus.preload entries must use ENGINE/MODEL selectors.";
      }
    ];

    environment.systemPackages = [ cfg.package ];

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];

    users.groups.chorus = { };
    users.users.chorus = {
      isSystemUser = true;
      group = "chorus";
      extraGroups = [
        "video"
        "render"
      ];
      description = "Chorus service user";
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDirectory} 0750 chorus chorus -"
      "d ${cfg.stateDirectory}/application 0750 chorus chorus -"
      "d ${cfg.cacheDirectory} 0750 chorus chorus -"
      "d ${cfg.modelsDirectory} 0750 chorus chorus -"
    ];

    systemd.services.chorus = {
      description = "Chorus multi-engine TTS API";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      path = [
        pkgs.uv
        pkgs.ffmpeg
        pkgs.git
        pkgs.python312
      ];
      environment = cfg.environment // protectedEnvironment;
      serviceConfig = {
        Type = "exec";
        User = "chorus";
        Group = "chorus";
        SupplementaryGroups = [
          "video"
          "render"
        ];
        ExecStart = utils.escapeSystemdExecArgs cliArgs;
        WorkingDirectory = cfg.stateDirectory;
        Restart = "on-failure";
        RestartSec = 10;
        TimeoutStartSec = "30min";
        TimeoutStopSec = 30;
        UMask = "0077";
        RuntimeDirectory = "chorus";
        RuntimeDirectoryMode = "0750";
        PrivateTmp = true;
        PrivateDevices = false;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        NoNewPrivileges = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        # Native Python wheels and JIT caches need executable mappings even
        # when the host marks writable service state noexec.
        ExecPaths = [
          "${cfg.stateDirectory}/application"
          cfg.cacheDirectory
        ];
        ReadWritePaths = [
          cfg.stateDirectory
          cfg.cacheDirectory
          cfg.modelsDirectory
        ];
        EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
      }
      // lib.optionalAttrs (cfg.stateDirectory == "/var/lib/chorus") {
        StateDirectory = "chorus";
        StateDirectoryMode = "0750";
      }
      // lib.optionalAttrs (cfg.cacheDirectory == "/var/cache/chorus") {
        CacheDirectory = "chorus";
        CacheDirectoryMode = "0750";
      };
    };
  };
}
