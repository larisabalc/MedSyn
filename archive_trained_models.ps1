# =======================
# Variables
# =======================

$FoldersToArchive = @(
    "diagnosis_engine\trained_models\context",
    "diagnosis_engine\trained_models\no_context"
)

$ArchiveDestination = "diagnosis_engine\trained_models"
$MaxSize = 100MB
$SevenZipPath = "C:\Program Files\7-Zip\7z.exe"

# =======================
# Validation
# =======================

if (-not (Test-Path $SevenZipPath)) {
    Write-Host "7-Zip not found at $SevenZipPath" -ForegroundColor Red
    exit 1
}

# =======================
# Archive each folder
# =======================

foreach ($Folder in $FoldersToArchive) {

    if (-not (Test-Path $Folder)) {
        Write-Host "Folder not found: $Folder" -ForegroundColor Red
        continue
    }

    $FolderName = Split-Path -Leaf $Folder
    $ArchiveName = Join-Path $ArchiveDestination "$FolderName.7z"

    Write-Host "Archiving: $Folder"
    Write-Host "Excluding: metrics"
    Write-Host "Archive: $ArchiveName"

    # 7-Zip arguments
    $arguments = @(
        "a",
        "$ArchiveName",
        "$Folder\*",
        "-xr!metrics",
        "-v$($MaxSize)",
        "-mx=9",
        "-y"
    )

    & "$SevenZipPath" @arguments

    # =======================
    # Post-archive cleanup
    # =======================

    if (Test-Path "$ArchiveName.001") {
        Write-Host "Archive created successfully." -ForegroundColor Green

        # Delete everything except metrics
        Get-ChildItem -Path $Folder -Force | Where-Object {
            $_.Name -ne "metrics"
        } | ForEach-Object {
            Remove-Item -Path $_.FullName -Recurse -Force
            Write-Host "Deleted: $($_.FullName)"
        }
    }
    else {
        Write-Host "Archive failed for $Folder" -ForegroundColor Red
    }

    Write-Host "----------------------------------------"
}

Write-Host "Archiving completed (metrics preserved)." -ForegroundColor Cyan
