<script lang="ts">
  import { onMount } from 'svelte';
  import ReplayResultCard from '$lib/components/ReplayResultCard.svelte';
  import { AntiCheatService } from '$lib/anticheat/AntiCheatService';
  import type { AntiCheatResult } from '$lib/types';
  import Alert from '$lib/components/Alert.svelte';

  let fileUpload: HTMLInputElement;
  let antiCheatResults: AntiCheatResult[] = [];
  let isFileProcessingOverlayVisible = false;
  let filesToProcessTotal = 0;
  let filesProcessed = 0;
  let fileCurrentlyProcessed = '';

  onMount(() => {
    // init ONNX runtime and load model
  });

  async function onInputFileChanged(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (!files) return;

    isFileProcessingOverlayVisible = true;
    filesToProcessTotal = files.length;
    filesProcessed = 0;
    antiCheatResults = [];

    for (let i = 0; i < files.length; i++) {
      fileCurrentlyProcessed = files[i].name;
      
      try {
        const result = await AntiCheatService.processReplay(files[i]);
        antiCheatResults = [...antiCheatResults, result];
      } catch (error) {
        console.error(`Error processing file ${files[i].name}:`, error);
      }

      filesProcessed++;
    }

    isFileProcessingOverlayVisible = false;
  }

  function clearUpload() {
    if (fileUpload) fileUpload.value = '';
    antiCheatResults = [];
  }
</script>

<svelte:head>
  <title>Anticheat - osu!Vendetta</title>
</svelte:head>

<Alert severity="warning">
  • This model is in its <b>ALPHA</b> phase with current accuracy around 93%. <br />
  • Do <b>NOT</b> rely solely on the model's output; double-check findings independently. <br />
</Alert>

<div class="file-upload">
  <input type="file" multiple bind:this={fileUpload} on:change={onInputFileChanged} accept=".osr" />
  <button on:click={clearUpload}>Clear</button>
</div>

{#if isFileProcessingOverlayVisible}
  <div class="processing-overlay">
    <p>Currently processing files</p>
    <p>{fileCurrentlyProcessed} ({filesProcessed}/{filesToProcessTotal})</p>
    <progress value={filesProcessed} max={filesToProcessTotal}></progress>
  </div>
{/if}

{#each antiCheatResults as result}
  <ReplayResultCard {result} />
{/each}

<style>
  .processing-overlay {
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
</style>