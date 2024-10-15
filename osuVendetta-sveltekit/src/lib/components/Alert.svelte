<script lang="ts">
  export let severity: 'normal' | 'warning' | 'error' = 'normal';
</script>

<div class="alert {severity}">
  <slot></slot>
</div>

<style>
  .alert {
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 4px;
  }
  .normal { background-color: #d1ecf1; color: #0c5460; }
  .warning { background-color: #fff3cd; color: #856404; }
  .error { background-color: #f8d7da; color: #721c24; }
</style>