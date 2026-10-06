# Sets up and starts a Minecraft Java Edition 26.3 server on Windows.
# - Installs Java 25 (Eclipse Temurin) if no Java 25+ is found
# - Downloads and verifies the server jar if it is missing
# - Accepts the Minecraft EULA (https://aka.ms/MinecraftEULA)
# - Creates start.bat and launches the server
#
# Run with:  powershell -ExecutionPolicy Bypass -File setup-minecraft-server.ps1

$ServerDir = "$HOME\minecraft-server"
$Jar       = "minecraft_server.26.3.jar"
$JarUrl    = "https://piston-data.mojang.com/v1/objects/33680f5f2ac32864d6d7cf5e56a705fdb3e05f4c/server.jar"
$JarSha1   = "33680F5F2AC32864D6D7CF5E56A705FDB3E05F4C"

function Find-Java25 {
    $roots = "C:\Program Files\Eclipse Adoptium", "C:\Program Files\Java", "C:\Program Files\Microsoft", "C:\Program Files\Zulu"
    $homes = @()
    foreach ($r in $roots) { if (Test-Path $r) { $homes += Get-ChildItem $r -Directory -ErrorAction SilentlyContinue } }
    $onPath = Get-Command java.exe -ErrorAction SilentlyContinue
    if ($onPath) { $homes += Get-Item (Split-Path (Split-Path $onPath.Source)) }
    foreach ($h in $homes) {
        $release = Join-Path $h.FullName "release"
        $exe = Join-Path $h.FullName "bin\java.exe"
        if ((Test-Path $release) -and (Test-Path $exe)) {
            if ((Get-Content $release -Raw) -match 'JAVA_VERSION="(\d+)') {
                if ([int]$Matches[1] -ge 25) { return $exe }
            }
        }
    }
    return $null
}

# 1. Java 25
$java = Find-Java25
if (-not $java) {
    Write-Host "Java 25 not found - installing Eclipse Temurin 25..." -ForegroundColor Cyan
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        # Exit code 1618 = another Windows install (often Windows Update) is running; wait and retry
        for ($i = 1; $i -le 10; $i++) {
            winget install --id EclipseAdoptium.Temurin.25.JDK -e --accept-source-agreements --accept-package-agreements
            if (Find-Java25) { break }
            Write-Host "Another install is busy - retrying in 60s ($i/10)..." -ForegroundColor Yellow
            Start-Sleep 60
        }
    } else {
        $msi = "$env:TEMP\temurin25.msi"
        curl.exe -L -o $msi "https://api.adoptium.net/v3/installer/latest/25/ga/windows/x64/jdk/hotspot/normal/eclipse"
        Start-Process msiexec.exe -Verb RunAs -Wait -ArgumentList "/i `"$msi`" /passive ADDLOCAL=FeatureMain,FeatureEnvironment,FeatureJarFileRunWith,FeatureJavaHome"
    }
    $java = Find-Java25
    if (-not $java) { Write-Host "Java 25 install failed. Install it manually from https://adoptium.net/ and re-run." -ForegroundColor Red; return }
}
Write-Host "Using Java: $java" -ForegroundColor Green

# 2. Server jar
New-Item -ItemType Directory -Force $ServerDir | Out-Null
Set-Location $ServerDir
if (-not (Test-Path $Jar) -or (Get-FileHash $Jar -Algorithm SHA1).Hash -ne $JarSha1) {
    Write-Host "Downloading server jar..." -ForegroundColor Cyan
    curl.exe -L -o $Jar $JarUrl
    if ((Get-FileHash $Jar -Algorithm SHA1).Hash -ne $JarSha1) { Write-Host "Server jar checksum mismatch." -ForegroundColor Red; return }
}

# 3. EULA
Set-Content -Path eula.txt -Value "eula=true" -Encoding ASCII
Write-Host "EULA accepted (https://aka.ms/MinecraftEULA)." -ForegroundColor Green

# 4. Memory: 4 GB if the PC has more than 8 GB of RAM, otherwise 2 GB
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$mem = if ($ramGB -gt 8) { "4G" } else { "2G" }
Write-Host "PC has $ramGB GB RAM - giving the server $mem." -ForegroundColor Green

# 5. start.bat for next time, then launch
Set-Content -Path start.bat -Encoding ASCII -Value "@echo off`r`ncd /d `"%~dp0`"`r`n`"$java`" -Xms$mem -Xmx$mem -jar $Jar nogui`r`npause"
Write-Host "Created $ServerDir\start.bat - double-click it to start the server in future." -ForegroundColor Green
Write-Host "Starting server... type 'stop' to shut it down." -ForegroundColor Cyan
& $java "-Xms$mem" "-Xmx$mem" -jar $Jar nogui
