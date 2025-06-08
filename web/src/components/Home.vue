<template>
  <!-- Camera View -->
  <div class="relative mx-4 pt-4 pt-[80px]">
    <div class="bg-white rounded-lg shadow-sm overflow-hidden">
      <div class="p-0">
        <div
          class="relative aspect-[4/3] bg-gradient-to-br from-gray-800 to-gray-900"
        >
          <img src="" alt="" srcset="" ref="imgResult" />
          <!-- Status indicators -->
          <div class="absolute top-3 left-3 flex gap-2">
            <span
              class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
            >
              <span class="text-green-500">●</span>
              &nbsp;LIVE
            </span>
            <span
              :class="[
                'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
                babyStatus === 'Cảnh báo'
                  ? 'bg-red-100 text-red-800'
                  : 'bg-blue-100 text-blue-800',
              ]"
            >
              {{ babyStatus }}
            </span>
          </div>

          <!-- Setting -->
          <!-- <div
            class="absolute top-3 right-3 z-50"
            @click="showSettingModal = true"
          >
            <Settings class="text-white" />
          </div> -->

          <!-- Detection overlay -->
          <div
            v-if="currentAlert"
            class="absolute inset-0 border-4 border-red-500 animate-pulse"
          />
        </div>
      </div>
    </div>
  </div>

  <!-- Status Card -->
  <div class="mx-4 mt-6">
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <CheckCircle
            v-if="babyStatus === 'An toàn'"
            class="h-8 w-8 text-green-500"
          />
          <AlertTriangle v-else class="h-8 w-8 text-red-500" />
          <div>
            <h3 class="font-semibold text-gray-800">
              {{ babyStatus === "An toàn" ? "Em bé an toàn" : "Cần chú ý" }}
            </h3>
            <p class="text-sm text-gray-600">
              {{ currentAlert || "Tư thế bình thường" }}
            </p>
          </div>
        </div>
        <div class="text-right">
          <div class="text-2xl font-bold text-gray-800">
            {{ babyStatus === "An toàn" ? "✓" : "!" }}
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="h-24"></div>

  <!-- Setting modal -->
  <!-- <div
    class="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
    v-if="showSettingModal"
  >
    <div class="bg-white rounded-lg p-6 m-4 w-full">
      <h3 class="text-lg font-semibold text-gray-800 mb-4">
        Cài đặt vị trí nôi
      </h3>

      <div
        class="relative aspect-[4/3] bg-gray-100 rounded-lg mb-4 overflow-hidden"
      >
        <div
          class="absolute inset-0 bg-gradient-to-br from-gray-300 to-gray-400"
        ></div>
      </div>

      <div class="flex gap-2 mb-5">
        <input
          type="number"
          placeholder="Nhập tọa độ phải"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />

        <input
          type="number"
          placeholder="Nhập tọa độ phải"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />

        <input
          type="number"
          placeholder="Nhập tọa độ phải"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />

        <input
          type="number"
          placeholder="Nhập tọa độ phải"
          class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      <button
        @click="saveCribPosition"
        class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors mb-5"
      >
        Tự động lấy vị trí
      </button>

      <div class="flex gap-3">
        <button
          @click="showSettingModal = false"
          class="flex-1 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          Hủy
        </button>
        <button
          @click="saveCribPosition"
          class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Lưu
        </button>
      </div>
    </div>
  </div> -->
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from "vue";
import { AlertTriangle, CheckCircle, Settings } from "lucide-vue-next";

const showSettingModal = ref(false);
const currentAlert = ref();
const babyStatus = ref("An toàn");
const imgResult = ref();

// Lifecycle hooks
onMounted(() => {
  const ws = new WebSocket("ws://103.70.12.120:8000/ws");
  ws.binaryType = "arraybuffer";

  ws.onmessage = function (event) {
    if (event.data instanceof ArrayBuffer) {
      const blob = new Blob([event.data], { type: "image/jpg" });
      const url = URL.createObjectURL(blob);
      imgResult.value.src = url;
    } else {
      const data = JSON.parse(event.data);

      if (data.baby_down_pose_result) {
        babyStatus.value = "Cảnh báo";
        currentAlert.value = "Em bé úp mặt";
      } else if (data.baby_not_in_crib_result) {
        babyStatus.value = "Cảnh báo";
        currentAlert.value = "Em bé không có trong nôi";
      } else if (data.unknown_person_result) {
        babyStatus.value = "Cảnh báo";
        currentAlert.value = "Có người lạ";
      } else {
        babyStatus.value = "An toàn";
        currentAlert.value = null;
      }
    }
  };
});
</script>

<style scoped>
.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.aspect-\[4\/3\] {
  aspect-ratio: 4 / 3;
}
</style>
