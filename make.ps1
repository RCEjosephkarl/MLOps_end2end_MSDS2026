# Windows PowerShell equivalent of the Makefile.
# Usage:  .\make.ps1 <target>
#   e.g.  .\make.ps1 build   ;  .\make.ps1 train   ;  .\make.ps1 serve
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

switch ($Target.ToLower()) {
    "build"  { docker compose build }
    "train"  { docker compose run --rm pipeline }
    "serve"  { docker compose up api gradio }
    "api"    { docker compose up api }
    "gradio" { docker compose up gradio }
    "mlflow" { docker compose --profile tools up mlflow }
    "test"   { docker compose run --rm test }
    "up"     { docker compose build; docker compose up api gradio }
    "down"   { docker compose down }
    "clean"  {
        docker compose down -v
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue reports\*, mlruns\*, models\*.joblib
    }
    default {
        Write-Host "Targets: build | train | serve | api | gradio | mlflow | test | up | down | clean"
        Write-Host "Example: .\make.ps1 train"
    }
}
