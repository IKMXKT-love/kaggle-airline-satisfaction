$ErrorActionPreference = 'Stop'
$cleanupRoot = 'C:\Users\ASUS\Documents\ChatGPT\kaggle_first'
$cleanupReview = Join-Path $cleanupRoot 'cleanup_review_20261022'
$cleanupManifestPath = Join-Path $cleanupReview 'proposed_deletion_manifest.json'
$cleanupManifest = Get-Content -LiteralPath $cleanupManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$cleanupPreflight = Get-Content -LiteralPath (Join-Path $cleanupReview 'A_preflight.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $cleanupManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $cleanupPreflight.manifest_sha256) {
    throw 'Deletion manifest changed after preflight.'
}
if ([System.IO.Path]::GetFullPath($cleanupManifest.root).TrimEnd('\') -ne $cleanupRoot) {
    throw 'Unexpected cleanup root.'
}
$cleanupDeleted = 0
$cleanupFailures = [System.Collections.Generic.List[object]]::new()
$cleanupLog = Join-Path $cleanupReview 'A_deletion_log.jsonl'
foreach ($cleanupGroup in $cleanupManifest.groups.PSObject.Properties) {
    if ($cleanupGroup.Name -notmatch '^A[1-5]_') { continue }
    foreach ($cleanupTarget in $cleanupGroup.Value) {
        try {
            $cleanupAbsolute = [System.IO.Path]::GetFullPath($cleanupTarget.path)
            $cleanupExpected = [System.IO.Path]::GetFullPath((Join-Path $cleanupRoot $cleanupTarget.relative_path))
            if ($cleanupAbsolute -ne $cleanupExpected -or -not $cleanupAbsolute.StartsWith($cleanupRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
                throw 'Target outside approved workspace.'
            }
            $cleanupItem = Get-Item -LiteralPath $cleanupAbsolute -Force
            if ($cleanupItem.PSIsContainer -or ($cleanupItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                throw 'Target is not an ordinary file.'
            }
            $cleanupAncestor = $cleanupItem.Directory
            while ($null -ne $cleanupAncestor -and $cleanupAncestor.FullName.StartsWith($cleanupRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                if ($cleanupAncestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Reparse-point ancestor.' }
                if ($cleanupAncestor.FullName -eq $cleanupRoot) { break }
                $cleanupAncestor = $cleanupAncestor.Parent
            }
            if ($cleanupItem.Length -ne $cleanupTarget.expected_bytes) { throw 'Target size changed.' }
            Remove-Item -LiteralPath $cleanupAbsolute -Force -ErrorAction Stop
            $cleanupDeleted++
            @{path=$cleanupAbsolute; bytes=$cleanupTarget.expected_bytes; group=$cleanupGroup.Name; status='deleted'} | ConvertTo-Json -Compress | Add-Content -LiteralPath $cleanupLog -Encoding utf8
        } catch {
            $cleanupFailure = @{path=$cleanupTarget.path; group=$cleanupGroup.Name; error=$_.Exception.Message}
            $cleanupFailures.Add($cleanupFailure)
            $cleanupFailure | ConvertTo-Json -Compress | Add-Content -LiteralPath $cleanupLog -Encoding utf8
        }
    }
    Write-Output ('Finished {0}; cumulative deleted files: {1}' -f $cleanupGroup.Name, $cleanupDeleted)
}
@{deleted_files=$cleanupDeleted; failures=@($cleanupFailures.ToArray())} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $cleanupReview 'A_deletion_result.json') -Encoding utf8
if ($cleanupFailures.Count -gt 0) { Write-Output ('Unremoved targets: {0}' -f $cleanupFailures.Count) }
