<script lang="ts">
  import type { AntiCheatResult } from '$lib/types';

  export let result: AntiCheatResult;
</script>

<div class="card">
  <h3>{result.fileName}</h3>
  <p>Player: {result.player}</p>
  <p>Status: <span class={result.status.toLowerCase()}>{result.status}</span></p>
  <p>Cheat Probability: {result.cheatProbability.toFixed(2)}%</p>
</div>

<style>
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