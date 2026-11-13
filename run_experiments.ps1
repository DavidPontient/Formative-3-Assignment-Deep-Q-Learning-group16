# Runs David's 10 experiments, one after another.
# Only gamma and batch size change. Everything else uses the defaults in train.py.
#
# Run all ten:      powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1
# Run only some:    powershell -ExecutionPolicy Bypass -File .\run_experiments.ps1 -Run 1,2,3

param([int[]]$Run = (1..10))

$TotalSteps = 250000
$Folder = "david_pontient"

$experiments = @(
    @{ Id = 1;  Gamma = "0.99";  Batch = "32"  },
    @{ Id = 2;  Gamma = "0.90";  Batch = "32"  },
    @{ Id = 3;  Gamma = "0.95";  Batch = "32"  },
    @{ Id = 4;  Gamma = "0.97";  Batch = "32"  },
    @{ Id = 5;  Gamma = "0.999"; Batch = "32"  },
    @{ Id = 6;  Gamma = "0.99";  Batch = "64"  },
    @{ Id = 7;  Gamma = "0.99";  Batch = "128" },
    @{ Id = 8;  Gamma = "0.95";  Batch = "64"  },
    @{ Id = 9;  Gamma = "0.999"; Batch = "64"  },
    @{ Id = 10; Gamma = "0.90";  Batch = "128" }
)

foreach ($e in $experiments) {
    if ($Run -notcontains $e.Id) { continue }
    $name = "exp{0:D2}" -f $e.Id
    Write-Host "Running $name with gamma $($e.Gamma) and batch size $($e.Batch)"
    uv run train.py --gamma $e.Gamma --batch_size $e.Batch --total_steps $TotalSteps --save_path "$Folder/$name.pth" --log_path "$Folder/$name.csv"
}
