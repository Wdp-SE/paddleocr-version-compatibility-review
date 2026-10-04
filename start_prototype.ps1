param(
    [int]$RagPort = 8765,
    [int]$UiPort = 8502,
    [string]$RagPythonPath = '',
    [switch]$EnableGeneration,
    [switch]$DisableGeneration
)

$ErrorActionPreference = 'Stop'
$workspace = $PSScriptRoot
$agentRoot = Join-Path $workspace 'change-review-agent'
$legacyAgentRoot = Join-Path $workspace 'OpenManus-rag'
$ragRoot = Join-Path $workspace 'versioned-rag-service'
$legacyRagRoot = Join-Path $workspace 'RAG-Challenge-2-main'
$uiRoot = Join-Path $workspace 'demo-ui'
$ragPython = Join-Path $ragRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython -PathType Leaf)) {
    # Keep the existing local virtual environment usable after the folder rename.
    $ragPython = Join-Path $legacyRagRoot '.venv\Scripts\python.exe'
}
if (-not [string]::IsNullOrWhiteSpace($RagPythonPath)) {
    $ragPython = (Resolve-Path -LiteralPath $RagPythonPath).Path
}
$uiPython = Join-Path $agentRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $uiPython -PathType Leaf)) {
    # Keep the existing local virtual environment usable after the folder rename.
    $uiPython = Join-Path $legacyAgentRoot '.venv\Scripts\python.exe'
}
$projectCorpus = Join-Path $ragRoot 'public_corpus_paddleocr'
$projectRetrievalPolicy = Join-Path $projectCorpus 'retrieval_policy.json'

# Codex can inject a loopback HTTP proxy into its child processes. If that
# proxy is unavailable while Windows has a configured system proxy, let the
# Python services fall back to the Windows proxy registry for this launch.
# This is process-scoped: the caller's environment is always restored.
$useWindowsProxyFallback = $false
$httpsProxy = @($env:HTTPS_PROXY, $env:https_proxy, $env:ALL_PROXY, $env:all_proxy) |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -First 1
if (-not [string]::IsNullOrWhiteSpace($httpsProxy)) {
    try {
        $proxyUri = [Uri]$httpsProxy
        $isLoopbackProxy = $proxyUri.IsLoopback -or $proxyUri.Host -eq 'localhost'
        if ($isLoopbackProxy) {
            $proxySocket = [Net.Sockets.TcpClient]::new()
            try {
                $connect = $proxySocket.ConnectAsync($proxyUri.Host, $proxyUri.Port)
                try {
                    $proxyReachable = $connect.Wait(750) -and $proxySocket.Connected
                } catch {
                    $proxyReachable = $false
                }
            } finally {
                $proxySocket.Dispose()
            }
            $windowsProxy = Get-ItemProperty `
                'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' `
                -ErrorAction SilentlyContinue
            $windowsProxyConfigured = $windowsProxy -and $windowsProxy.ProxyEnable -eq 1 `
                -and -not [string]::IsNullOrWhiteSpace($windowsProxy.ProxyServer)
            $useWindowsProxyFallback = -not $proxyReachable -and $windowsProxyConfigured
        }
    } catch {
        # Keep explicit proxy settings when their status cannot be determined.
        $useWindowsProxyFallback = $false
    }
}

function Start-PrototypeProcess([hashtable]$StartParameters) {
    if (-not $useWindowsProxyFallback) {
        return Start-Process @StartParameters
    }

    $proxyNames = @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    $savedProxyEnvironment = @{}
    foreach ($name in $proxyNames) {
        $savedProxyEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
    try {
        return Start-Process @StartParameters
    } finally {
        foreach ($name in $proxyNames) {
            $value = $savedProxyEnvironment[$name]
            if ($null -eq $value) {
                Remove-Item "Env:$name" -ErrorAction SilentlyContinue
            } else {
                [Environment]::SetEnvironmentVariable($name, $value, 'Process')
            }
        }
    }
}

foreach ($required in @($ragPython, $uiPython, $projectRetrievalPolicy)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "缺少启动依赖：$required；请先按 README 安装环境。"
    }
}

function Test-Endpoint([string]$Uri) {
    try {
        Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

$ragUrl = "http://127.0.0.1:$RagPort"
$uiUrl = "http://127.0.0.1:$UiPort"
if (Test-Endpoint "$ragUrl/health") {
    throw "RAG 端口 $RagPort 已被占用；请停止旧服务或指定 -RagPort。"
}
if (Test-Endpoint $uiUrl) {
    throw "UI 端口 $UiPort 已被占用；请停止旧服务或指定 -UiPort。"
}

$runRoot = Join-Path $env:TEMP ('rag-agent-public-demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null

$env:APP_ENV = 'public_demo'
$env:RD_V2_PROJECT_ROOT = $ragRoot
$env:RAG_PUBLIC_CORPUS_ROOT = $projectCorpus
$env:RAG_PUBLIC_RETRIEVAL_CONFIG = Join-Path $projectCorpus 'public_retrieval_runtime.json'
if ($EnableGeneration -and $DisableGeneration) {
    throw '不能同时指定 -EnableGeneration 和 -DisableGeneration。'
}
$provider = if ([string]::IsNullOrWhiteSpace($env:RD_V2_GENERATION_PROVIDER)) {
    'deepseek'
} else {
    $env:RD_V2_GENERATION_PROVIDER.Trim().ToLowerInvariant()
}
$providerKeyName = switch ($provider) {
    'dashscope' { 'DASHSCOPE_API_KEY' }
    'deepseek' { 'DEEPSEEK_API_KEY' }
    default { $null }
}
$providerKeyConfigured = $providerKeyName -and
    -not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($providerKeyName, 'Process'))
$generationEnabled = -not $DisableGeneration -and ($EnableGeneration -or $providerKeyConfigured)
if ($generationEnabled) {
    switch ($provider) {
        'dashscope' {
            $apiKeyName = 'DASHSCOPE_API_KEY'
            $defaultModel = 'qwen-turbo'
        }
        'deepseek' {
            $apiKeyName = 'DEEPSEEK_API_KEY'
            $defaultModel = 'deepseek-flash'
        }
        default {
            throw 'RD_V2_GENERATION_PROVIDER 仅支持 dashscope 或 deepseek。'
        }
    }
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($apiKeyName, 'Process'))) {
        throw "启用 $provider 生成前，请先在本机环境变量设置 $apiKeyName。"
    }
    $env:RD_V2_GENERATION_PROVIDER = $provider
    if ([string]::IsNullOrWhiteSpace($env:RD_V2_GENERATION_MODEL)) {
        $env:RD_V2_GENERATION_MODEL = $defaultModel
    }
}
$env:RD_V2_ALLOW_EXTERNAL_GENERATION = if ($generationEnabled) { 'true' } else { 'false' }

if ($useWindowsProxyFallback) {
    Write-Host '检测到不可用的本机代理；RAG/UI 子进程本次回退到 Windows 系统代理，不修改永久环境变量。'
}
$ragProcess = Start-PrototypeProcess @{ FilePath = $ragPython; ArgumentList = @('-m', 'uvicorn', 'src.public_server:app', '--host', '127.0.0.1', '--port', "$RagPort"); WorkingDirectory = $ragRoot; RedirectStandardOutput = (Join-Path $runRoot 'rag.stdout.log'); RedirectStandardError = (Join-Path $runRoot 'rag.stderr.log'); WindowStyle = 'Hidden'; PassThru = $true }

$ragReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-Endpoint "$ragUrl/health") {
        $ragReady = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $ragReady) {
    throw "RAG 服务未能启动，请查看 $runRoot\rag.stderr.log"
}

$env:RAG_API_BASE_URL = $ragUrl
$env:DEMO_RUNTIME_ROOT = Join-Path $runRoot 'ui-runtime'
$env:DEMO_DATA_CLASSIFICATION = 'Official Public'
$env:DEMO_ALLOW_RAG_QUERY = if ($generationEnabled) { 'true' } else { 'false' }
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
$env:STREAMLIT_SERVER_HEADLESS = 'true'
$uiProcess = Start-PrototypeProcess @{ FilePath = $uiPython; ArgumentList = @('-m', 'streamlit', 'run', 'app.py', '--server.address', '127.0.0.1', '--server.port', "$UiPort", '--server.headless', 'true', '--browser.gatherUsageStats', 'false'); WorkingDirectory = $uiRoot; RedirectStandardOutput = (Join-Path $runRoot 'ui.stdout.log'); RedirectStandardError = (Join-Path $runRoot 'ui.stderr.log'); WindowStyle = 'Hidden'; PassThru = $true }

$uiReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if (Test-Endpoint $uiUrl) {
        $uiReady = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $uiReady) {
    throw "UI 未能启动，请查看 $runRoot\ui.stderr.log"
}

Write-Host "官方公开研发资料工作台已启动：$uiUrl"
Write-Host "RAG API：$ragUrl/docs"
Write-Host "临时运行目录：$runRoot"
Write-Host "进程 ID：RAG $($ragProcess.Id)，UI $($uiProcess.Id)"
