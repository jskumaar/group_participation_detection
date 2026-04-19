# # record_all_luxonis_altname.ps1
# # Automatically record from all Luxonis UVC Cameras using their alternative names (Windows)

# $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
# $outputDir = "recordings"
# $fps = 30
# $res = "1920x1080"

# if (!(Test-Path $outputDir)) {
#     New-Item -ItemType Directory -Path $outputDir | Out-Null
# }

# Write-Host "Detecting Luxonis UVC cameras by alternative name..."

# # Run ffmpeg and collect lines for Luxonis devices
# $altNames = @()
# $lines = ffmpeg -list_devices true -f dshow -i dummy 2>&1
# for ($i = 0; $i -lt $lines.Count; $i++) {
#     $line = $lines[$i]
#     if ($line -match '"Luxonis UVC Camera"') {
#         # The next line after the camera name contains its "Alternative name"
#         $nextLine = $lines[$i + 1]
#         if ($nextLine -match 'Alternative name "([^"]+)"') {
#             $altNames += $matches[1]
#         }
#     }
# }

# if ($altNames.Count -eq 0) {
#     Write-Host "❌ No Luxonis cameras found."
#     exit
# }

# Write-Host ""
# Write-Host "🎥 Found $($altNames.Count) Luxonis camera(s):"
# $altNames | ForEach-Object { Write-Host " - $_" }
# Write-Host ""

# # Start recording each one
# $i = 0
# foreach ($alt in $altNames) {
#     $outFile = "$outputDir\cam${i}_$timestamp.mp4"
#     Write-Host "▶ Starting recording for camera $i -> $outFile"
#     Start-Process -NoNewWindow ffmpeg `
#         -ArgumentList @(
#             "-f", "dshow",
#             "-framerate", "$fps",
#             "-video_size", "$res",
#             "-i", "video=$alt",
#             "-c:v", "libx264",
#             "-preset", "ultrafast",
#             "-pix_fmt", "yuv420p",
#             "$outFile"
#         )
#     $i++
# }

# Write-Host ""
# Write-Host "✅ All Luxonis cameras are now recording."
# Write-Host "Press Ctrl+C in each ffmpeg window to stop."



##################


# Requires FFmpeg to be installed and available in the system PATH.

# Get the DirectShow device list from FFmpeg
Write-Host "Searching for available video devices..."
$ffmpegOutput = ffmpeg.exe -list_devices true -f dshow -i dummy 2>&1

# Find the name of the first video device
$videoDeviceName = ($ffmpegOutput | Select-String -Pattern '"(.*?)"' | Where-Object { $_.Line -like '*video devices*' } | Select-Object -First 1).ToString().Split('"')[1]

# Check if a video device was found
if (-not $videoDeviceName) {
    Write-Error "No video devices found. Please check your camera connections."
    return
}

# Prompt user to confirm device and start recording
Write-Host "Found video device: '$videoDeviceName'"
$outputFile = "output-recording-$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').mp4"

# FFmpeg recording command
$ffmpegCommand = "& ffmpeg.exe -f dshow -i video=`"$videoDeviceName`" -c:v libx264 -preset veryfast -crf 23 -t 60 `"$outputFile`""

# Display the recording command and start recording
Write-Host "Starting recording from '$videoDeviceName' for 60 seconds. Output will be saved to '$outputFile'."
Write-Host "Press Ctrl+C to stop recording early."
Write-Host ""
Invoke-Expression $ffmpegCommand

Write-Host ""
Write-Host "Recording finished."
