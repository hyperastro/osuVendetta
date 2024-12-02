<script lang="ts">
  // the UI is not pretty yet
  
  import { onMount } from 'svelte';
  import Alert from '$lib/components/Alert.svelte';
  import { drawerOpen, isDarkMode, processReplay, type AntiCheatResult, type Severity } from '../lib/utils';

  let antiCheatResults: AntiCheatResult[] = [];
  let isProcessing = false;
  let progress = { total: 0, processed: 0, current: '' };
  let isModelLoaded = false;

  onMount(async () => {
    try {
      await initializeModel();
      isModelLoaded = true;
    } catch (error) {
      console.error('Error initializing ONNX model:', error);
    }
  });

  const initializeModel = async () => {
    // TODO (skyfly): initalize ONNX model
    await new Promise(resolve => setTimeout(resolve, 1000));
  };

  const processFiles = async (files: FileList) => {
    isProcessing = true;
    progress = { total: files.length, processed: 0, current: '' };
    antiCheatResults = [];

    for (const file of Array.from(files)) {
      progress.current = file.name;
      try {
        const result = await processReplay(file);
        antiCheatResults = [...antiCheatResults, result];
      } catch (error) {
        console.error(`Error processing file ${file.name}:`, error);
      }
      progress.processed++;
    }

    isProcessing = false;
  };

  const handleFileSelect = (event: Event) => processFiles((event.target as HTMLInputElement).files);
  const handleClear = () => {
    antiCheatResults = [];
    (document.getElementById('file-upload') as HTMLInputElement).value = '';
  };
</script>

<svelte:head>
  <title>osu!Vendetta - AI Powered Anti-Cheat</title>
</svelte:head>

<div class={$isDarkMode ? 'dark-mode' : 'light-mode'}>
  <header>
    <h1>osu!Vendetta</h1>
    // <img src="/logo.png" alt="Logo" height="34" width="60" />
	// TODO (hallow): implement dark mode functionality & proper drawer
    <button on:click={() => $drawerOpen = !$drawerOpen}>Toggle Drawer</button>
    <button on:click={() => $isDarkMode = !$isDarkMode}>Toggle Dark Mode</button>
  </header>

  <div class="layout">
    {#if $drawerOpen}
      <nav>
        <a href="#home" class:active={true}>Home</a>
        <a href="#anticheat" class:active={false}>Anticheat</a>
      </nav>
    {/if}
    <main>
      <h1>osu!Vendetta</h1>
      <p>AI Powered Anti-Cheat for osu!</p>

      <Alert severity="warning">
        <p>• This project is currently in its <strong>ALPHA</strong> phase.</p>
        <p>• If you find any bugs please report them on Github or via Discord.</p>
      </Alert>

      <Alert severity="normal">
        <p>
          You can find the project here:
          <a href="https://github.com/160IQAstro/osu-AI-Anti-Cheat-Project" target="_blank" rel="noopener noreferrer">
            github.com/160IQAstro/osu-AI-Anti-Cheat-Project
          </a>
        </p>
      </Alert>

      <Alert severity="normal">
        <p>
          Join our discord:
          <a href="https://discord.gg/BDU3W22HEW" target="_blank" rel="noopener noreferrer">
            discord.gg/BDU3W22HEW
          </a>
        </p>
      </Alert>

      <h2>Anticheat</h2>

      <Alert severity="warning">
        • This model is in its <b>ALPHA</b> phase with current accuracy around 93%. <br />
        • Do <b>NOT</b> rely solely on the model's output; double-check findings independently. <br />
      </Alert>

      {#if !isModelLoaded}
        <div class="overlay">Loading ONNX model...</div>
      {:else}
        <div class="file-upload">
          <input id="file-upload" type="file" multiple accept=".osr" on:change={handleFileSelect} />
          <button on:click={handleClear}>Clear</button>
        </div>

        {#if isProcessing}
          <div class="overlay">
            <p>Currently processing files</p>
            <p>{progress.current} ({progress.processed}/{progress.total})</p>
            <progress value={progress.processed} max={progress.total}></progress>
          </div>
        {/if}

        {#each antiCheatResults as result (result.fileName)}
          <div class="card">
            <h3>{result.fileName}</h3>
            <p>Player: {result.player}</p>
            <p>Status: <span class={result.status.toLowerCase()}>{result.status}</span></p>
            <p>Cheat Probability: {result.cheatProbability.toFixed(2)}%</p>
          </div>
        {/each}
      {/if}
    </main>
  </div>
</div>

<style>
  .overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    flex-direction: column;
	justify-content: center;
    align-items: center;
    color: white;
  }
  .card {
    border: 1px solid #ddd;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 4px;
  }
  .normal { color: green; }
  .suspicious { color: orange; }
  .cheating { color: red; }
</style>