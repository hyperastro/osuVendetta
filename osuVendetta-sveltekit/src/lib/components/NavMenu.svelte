<script lang="ts">
  import { page } from '$app/stores';
</script>

<nav>
  <a href="/" class:active={$page.url.pathname === '/'}>Home</a>
  <a href="/anticheat" class:active={$page.url.pathname === '/anticheat'}>Anticheat</a>
</nav>

<style>
  /* styles */
</style>