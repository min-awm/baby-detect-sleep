<template>
  <div class="mx-auto bg-gradient-to-b from-blue-50 to-indigo-50 min-h-screen">
    <!-- Header -->
    <div
      class="flex items-center justify-between p-4 bg-white backdrop-blur-sm fixed top-0 left-0 right-0 z-1 border-gray-200 border-b"
    >
      <div class="flex items-center gap-2">
        <h1 class="font-semibold text-gray-800">{{ title }}</h1>
      </div>

      <div class="flex items-center gap-2">
        <a href="">
          <RefreshCw />
        </a>
      </div>
    </div>

    <Home v-if="page == 0" />
    <Alert v-if="page == 1" />

    <!-- Bottom Navigation -->
    <!-- <div
      class="fixed bottom-0 left-0 right-0 bg-white/90 backdrop-blur-sm border-t border-gray-200"
    >
      <div class="grid grid-cols-2 gap-1 p-2">
        <button
          class="flex flex-col items-center justify-center h-12 gap-1 hover:bg-gray-100 rounded transition-colors"
          @click="page = 0"
        >
          <Camera :class="{ 'h-4 w-4': true, 'text-blue-600': page == 0 }" />
          <span :class="{ 'text-xs': true, 'text-blue-600': page == 0 }"
            >Giám sát</span
          >
        </button>
        <button
          class="flex flex-col items-center justify-center h-12 gap-1 hover:bg-gray-100 rounded transition-colors"
          @click="page = 1"
        >
          <Bell :class="{ 'h-4 w-4': true, 'text-blue-600': page == 1 }" />
          <span :class="{ 'text-xs': true, 'text-blue-600': page == 1 }"
            >Cảnh báo</span
          >
        </button>
      </div>
    </div> -->
  </div>
</template>

<script setup>
import { ref, computed } from "vue";
import { RefreshCw } from "lucide-vue-next";
import Home from "./components/Home.vue";
import Alert from "./components/Alert.vue";

const page = ref(0);
const title = computed(() => {
  return page.value === 0 ? "Baby sleep" : "Cảnh báo";
});
</script>
