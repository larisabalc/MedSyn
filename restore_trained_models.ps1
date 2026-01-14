# =======================
# Variables
# =======================

$ArchivesToExtract = @(
    "diagnosis_engine\trained_models\context.7z.001",
    "diagnosis_engine\trained_models\no_context.7z.001"
)

$BaseDir = "diagnosis_engine\trained_models"
$SevenZipPath = "C:\Program Files\7-Zip\7z.exe"

# =======================
# Validation
# =======================

if (-not (Test-Path $SevenZipPath)) {
    Write-Host "7-Zip not found at $SevenZipPath" -ForegroundColor Red
    exit 1
}

# =======================
# Extract archives
# =======================

foreach ($Archive in $ArchivesToExtract) {

    if (-not (Test-Path $Archive)) {
        Write-Host "Archive not found: $Archive" -ForegroundColor Red
        continue
    }

    $ArchiveFile = Split-Path -Leaf $Archive
    $ModelName = $ArchiveFile -replace "\.7z\.001$", ""
    $DestinationFolder = Join-Path $BaseDir $ModelName
    $TempExtractFolder = Join-Path $DestinationFolder "_tmp_extract"

    Write-Host "Restoring: $ModelName"

    # Ensure destination exists
    if (-not (Test-Path $DestinationFolder)) {
        New-Item -Path $DestinationFolder -ItemType Directory | Out-Null
    }

    # Create temp folder
    New-Item -Path $TempExtractFolder -ItemType Directory -Force | Out-Null

    # Extract to temp folder
    & "$SevenZipPath" x `
        "$Archive" `
        "-o$TempExtractFolder" `
        -y

    # Detect nested archive structure and move only real contents
    $NestedPath = Join-Path $TempExtractFolder "diagnosis_engine\trained_models\$ModelName"

    if (Test-Path $NestedPath) {
        $ItemsToMove = Get-ChildItem -Path $NestedPath -Force
    } else {
        $ItemsToMove = Get-ChildItem -Path $TempExtractFolder -Force
    }

    foreach ($Item in $ItemsToMove) {
        if ($Item.Name -ne "metrics") {
            Move-Item -Path $Item.FullName -Destination $DestinationFolder -Force
        }
    }

    # Cleanup temp folder
    Remove-Item -Path $TempExtractFolder -Recurse -Force

    # =======================
    # Delete archive parts
    # =======================

    $ArchiveBase = Join-Path $BaseDir "$ModelName.7z"
    Get-ChildItem "$ArchiveBase.*" | ForEach-Object {
        Remove-Item -Path $_.FullName -Force
        Write-Host "Deleted archive: $($_.Name)"
    }

    Write-Host "Restored and cleaned up: $ModelName" -ForegroundColor Green
    Write-Host "----------------------------------------"
}

Write-Host "Restore completed. Archives removed." -ForegroundColor Cyan
