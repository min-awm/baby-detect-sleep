<template>
  <div class="pt-[80px]">
    <div class="mx-4 mb-20 bg-white/70 rounded-lg shadow-sm">
      <div class="p-4">
        <h3 class="font-semibold text-gray-800 mb-3">Cảnh báo gần đây</h3>
        <div class="space-y-2">
          <div
            v-for="alert in recentAlerts"
            :key="alert.id"
            class="flex items-center justify-between text-sm"
          >
            <span class="text-gray-600">{{ alert.message }}</span>
            <span class="text-gray-400">{{ alert.time }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";


const recentAlerts = ref([
  { id: 1, message: "Em bé nằm nghiêng", time: "2 phút trước" },
  { id: 2, message: "Em bé không có trong cũi", time: "15 phút trước" },
  { id: 3, message: "Em bé úp mặt", time: "1 giờ trước" },
]);
</script>
