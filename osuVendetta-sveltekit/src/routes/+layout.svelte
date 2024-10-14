<script lang="ts">
  import { page } from '$app/stores';
  import NavMenu from '$lib/components/NavMenu.svelte';
  import '../app.css';

  let drawerOpen = true;
  let isDarkMode = true;

  function toggleDrawer() {
    drawerOpen = !drawerOpen;
  }

  function toggleDarkMode() {
    isDarkMode = !isDarkMode;
  }
</script>

<div class={isDarkMode ? 'dark-mode' : 'light-mode'}>
  <header>
    <h1>osu!Vendetta</h1>
    <img src="/logo.png" alt="Logo" height="34" width="60" />
    <button on:click={toggleDrawer}>Toggle Drawer</button>
    <button on:click={toggleDarkMode}>Toggle Dark Mode</button>
  </header>

  <div class="layout">
    {#if drawerOpen}
      <NavMenu />
    {/if}
    <main>
      <slot />
    </main>
  </div>
</div>

<style>
  /* styles */
</style>