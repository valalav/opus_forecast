param(
  [string]$SourcePath = $PSScriptRoot,
  [int]$Port = 8765,
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$AppRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$DefaultSource = [IO.Path]::GetFullPath($SourcePath)
$TokenBytes = New-Object byte[] 32
$Rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$Rng.GetBytes($TokenBytes)
$Rng.Dispose()
$Token = [Convert]::ToBase64String($TokenBytes).TrimEnd('=').Replace('+','-').Replace('/','_')
$Origin = "http://127.0.0.1:$Port"
$Listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
try { $Listener.Start() } catch { throw "Cannot bind local server to $Origin`: $($_.Exception.Message)" }

# TcpListener avoids Windows HTTP.sys URL ACL/admin requirements.
function Send-Response($Context, [int]$Status, [string]$ContentType, [byte[]]$Bytes) {
  $reason = if ($Status -eq 200) {'OK'} else {'Error'}
  $header = "HTTP/1.1 $Status $reason`r`nContent-Type: $ContentType`r`nContent-Length: $($Bytes.Length)`r`nConnection: close`r`nCache-Control: no-store`r`nX-Content-Type-Options: nosniff`r`n`r`n"
  $headerBytes = [Text.Encoding]::ASCII.GetBytes($header)
  try {
    $Context.Stream.Write($headerBytes,0,$headerBytes.Length)
    if ($Bytes.Length) { $Context.Stream.Write($Bytes,0,$Bytes.Length) }
    $Context.Stream.Flush()
  } finally { $Context.Client.Close() }
}
function Read-Context($Client) {
  $stream=$Client.GetStream(); $stream.ReadTimeout=2000; $stream.WriteTimeout=30000
  $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$false,1024,$true)
  $line=$reader.ReadLine()
  if (-not $line -or $line.Length -gt 4096) {throw 'Invalid request line'}
  $parts=$line.Split(' ')
  if ($parts.Length -ne 3 -or -not $parts[1].StartsWith('/') -or $parts[1].StartsWith('//')) {throw 'Invalid request'}
  $headers=@{}; $length=$line.Length
  while ($true) {
    $line=$reader.ReadLine(); if ($null -eq $line) {throw 'Incomplete headers'}
    $length += $line.Length; if ($length -gt 16384) {throw 'Headers too large'}
    if ($line -eq '') {break}
    $colon=$line.IndexOf(':'); if ($colon -lt 1) {throw 'Invalid header'}
    $name=$line.Substring(0,$colon).Trim()
    if ($headers.ContainsKey($name)) {throw 'Duplicate header'}
    $headers[$name]=$line.Substring($colon+1).Trim()
  }
  $uri=[Uri]::new("$Origin$($parts[1])")
  $query=[System.Web.HttpUtility]::ParseQueryString($uri.Query)
  return [pscustomobject]@{Client=$Client;Stream=$stream;Request=[pscustomobject]@{Headers=$headers;Url=$uri;QueryString=$query;HttpMethod=$parts[0]}}
}
function Json-Bytes($Value) { [Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 8 -Compress)) }
function Is-AllowedLocalFile([string]$Name) {
  return ($Name -ceq 'regional_indices.json' -or $Name -ceq 'macro_rates.json' -or $Name -ceq 'inflation_data.csv' -or $Name -match '^(?:\d{4}-\d{2}\s+)?ИПЦ с исключением сезонности.*регионы.*\.xlsx$')
}
function Is-UnsupportedWorkbook([string]$Name) { return ($Name -match '^ИПЦ полный.*\.xlsx$') }

Write-Host "Локальный сервер: $Origin/regional-inflation.html"
Write-Host "Папка данных по умолчанию: $DefaultSource"
Write-Host 'Остановить сервер: Ctrl+C'
if (-not $NoBrowser) { Start-Process "$Origin/regional-inflation.html" }
try {
  while ($true) {
    $Client = $Listener.AcceptTcpClient()
    try { $Context = Read-Context $Client } catch { $Client.Close(); continue }
    try {
      $Request = $Context.Request
      $HostHeader = $Request.Headers['Host']
      $RequestOrigin = $Request.Headers['Origin']
      if ($HostHeader -ne "127.0.0.1:$Port" -or ($RequestOrigin -and $RequestOrigin -ne $Origin)) {
        Send-Response $Context 403 'text/plain; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes('Forbidden')); continue
      }
      $route = [Uri]::UnescapeDataString($Request.Url.AbsolutePath)
      if ($route.StartsWith('/api/')) {
        if ($Request.HttpMethod -ne 'GET' -or $Request.Headers['X-Local-Token'] -cne $Token) {
          Send-Response $Context 403 'application/json; charset=utf-8' (Json-Bytes @{error='forbidden'}); continue
        }
        if ($route -eq '/api/cbr-rates') {
          $series = $Request.QueryString['series']
          $fromText = $Request.QueryString['from']
          $toText = $Request.QueryString['to']
          if ($series -notin @('ki','ruonia') -or $fromText -notmatch '^\d{4}-\d{2}-\d{2}$' -or $toText -notmatch '^\d{4}-\d{2}-\d{2}$') {
            Send-Response $Context 400 'application/json; charset=utf-8' (Json-Bytes @{error='invalid_parameters'}); continue
          }
          try { $fromDate = [DateTime]::ParseExact($fromText,'yyyy-MM-dd',[Globalization.CultureInfo]::InvariantCulture); $toDate = [DateTime]::ParseExact($toText,'yyyy-MM-dd',[Globalization.CultureInfo]::InvariantCulture) } catch {
            Send-Response $Context 400 'application/json; charset=utf-8' (Json-Bytes @{error='invalid_date'}); continue
          }
          if ($fromDate -gt $toDate -or ($toDate - $fromDate).TotalDays -gt 15000) { Send-Response $Context 400 'application/json; charset=utf-8' (Json-Bytes @{error='invalid_range'}); continue }
          $operation = if ($series -eq 'ki') {'KeyRateXML'} else {'RuoniaXML'}
          $soapAction = "http://web.cbr.ru/$operation"
          $body = "<soap:Envelope xmlns:soap=`"http://schemas.xmlsoap.org/soap/envelope/`"><soap:Body><$operation xmlns=`"http://web.cbr.ru/`"><fromDate>$($fromDate.ToString('yyyy-MM-dd'))T00:00:00</fromDate><ToDate>$($toDate.ToString('yyyy-MM-dd'))T00:00:00</ToDate></$operation></soap:Body></soap:Envelope>"
          try {
            $upstream = Invoke-WebRequest -Uri 'https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx' -Method Post -ContentType 'text/xml; charset=utf-8' -Headers @{SOAPAction=$soapAction} -Body $body -UseBasicParsing -TimeoutSec 30
            Send-Response $Context 200 'application/xml; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes($upstream.Content)); continue
          } catch { Send-Response $Context 502 'application/json; charset=utf-8' (Json-Bytes @{error='cbr_unavailable';message=$_.Exception.Message}); continue }
        }
        $rawPath = $Request.QueryString['path']
        if (-not $rawPath) { $rawPath = '.' }
        if ($route -eq '/api/local-sources') {
          if ($rawPath -eq '.') { $dir = $DefaultSource } else { $dir = if ([IO.Path]::IsPathRooted($rawPath)) {[IO.Path]::GetFullPath($rawPath)} else {[IO.Path]::GetFullPath((Join-Path $DefaultSource $rawPath))} }
          if (-not [IO.Directory]::Exists($dir)) { Send-Response $Context 404 'application/json; charset=utf-8' (Json-Bytes @{error='directory_not_found'}); continue }
          $items = @([IO.Directory]::EnumerateFiles($dir) | ForEach-Object { $info = [IO.FileInfo]$_; if ((Is-AllowedLocalFile $info.Name) -or (Is-UnsupportedWorkbook $info.Name)) { @{name=$info.Name;size=$info.Length;modified=$info.LastWriteTimeUtc.ToString('o');supported=(Is-AllowedLocalFile $info.Name)} } })
          Send-Response $Context 200 'application/json; charset=utf-8' (Json-Bytes @{available=$true;mode='loopback-folder';files=$items;defaultPath=$DefaultSource}); continue
        }
        if ($route -eq '/api/local-file') {
          $full = if ([IO.Path]::IsPathRooted($rawPath)) {[IO.Path]::GetFullPath($rawPath)} else {[IO.Path]::GetFullPath((Join-Path $DefaultSource $rawPath))}
          $name = [IO.Path]::GetFileName($full)
          if (-not (Is-AllowedLocalFile $name) -or -not [IO.File]::Exists($full)) { Send-Response $Context 404 'application/json; charset=utf-8' (Json-Bytes @{error='file_not_found'}); continue }
          $info = [IO.FileInfo]$full
          if ($info.Length -gt 209715200) { Send-Response $Context 413 'text/plain; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes('File exceeds 200 MB')); continue }
          $type = if ($name.EndsWith('.xlsx')) {'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'} elseif ($name.EndsWith('.csv')) {'text/csv; charset=utf-8'} else {'application/json; charset=utf-8'}
          Send-Response $Context 200 $type ([IO.File]::ReadAllBytes($full)); continue
        }
        Send-Response $Context 404 'application/json; charset=utf-8' (Json-Bytes @{error='not_found'}); continue
      }
      if ($Request.HttpMethod -ne 'GET') { Send-Response $Context 405 'text/plain' ([Text.Encoding]::UTF8.GetBytes('Method not allowed')); continue }
      $relativeSafe = $route.TrimStart('/').Replace('\\','/')
      if (-not $relativeSafe) { $relativeSafe = 'regional-inflation.html' }
      $staticAllow = @('regional-inflation.html','web/index.html','web/style.css','web/app.js','web/source-import.js','web/local-sources.js','web/cbr-rates.js','core/pkg/regional_inflation_core.js','core/pkg/regional_inflation_core_bg.wasm','data/regional_indices.json')
      if ($relativeSafe -notin $staticAllow) { Send-Response $Context 404 'text/plain; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes('Not found')); continue }
      $fullStatic = Join-Path $AppRoot $relativeSafe
      if (-not [IO.File]::Exists($fullStatic)) { Send-Response $Context 404 'text/plain; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes('Not found')); continue }
      $ext = [IO.Path]::GetExtension($fullStatic).ToLowerInvariant()
      if ($ext -notin @('.html','.css','.js','.wasm','.json')) { Send-Response $Context 403 'text/plain' ([Text.Encoding]::UTF8.GetBytes('Forbidden')); continue }
      $bytes = [IO.File]::ReadAllBytes($fullStatic)
      if ($ext -eq '.html') {
        $html = [Text.Encoding]::UTF8.GetString($bytes)
        $config = '<script>window.LOCAL_SOURCES_CONFIG=' + (ConvertTo-Json -InputObject @{baseUrl=$Origin;token=$Token;defaultPath=$DefaultSource} -Compress) + ';</script>'
        $html = $html.Replace('</head>', "$config</head>")
        $bytes = [Text.Encoding]::UTF8.GetBytes($html)
      }
      $mime = switch ($ext) { '.html' {'text/html; charset=utf-8'} '.css' {'text/css; charset=utf-8'} '.js' {'text/javascript; charset=utf-8'} '.wasm' {'application/wasm'} '.json' {'application/json; charset=utf-8'} }
      Send-Response $Context 200 $mime $bytes
    } catch {
      try { Send-Response $Context 500 'text/plain; charset=utf-8' ([Text.Encoding]::UTF8.GetBytes('Local server error')) } catch {}
    }
  }
} finally { $Listener.Stop() }
