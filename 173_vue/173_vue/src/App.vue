<script setup>
import {RouterLink, RouterView} from 'vue-router'
import menuTab from './components/menuTab.vue'
import MyHeader from "@/components/myHeader.vue";
import MyFooter from "@/components/myFooter.vue";
</script>

<template>
  <div class="container1">
    <div class="header">
      <my-header></my-header>
      <menu-tab></menu-tab>
    </div>
    <div class="mid">
      <RouterView/>
    </div>
    <div class="footer">
      <my-footer></my-footer>
    </div>
  </div>
</template>

<style scoped>

.container1 {
  display: flex;
  flex-direction: column;
}

.header {
  height: 120px;
}

.mid {
  overflow: auto;
  height: calc(100vh - 120px - 120px);
}

.footer {
  height: 120px;
}

html, body, #app {
  height: 100%;
  width: 100%;
  padding: 0;
  margin: 0 !important;
}
</style>
