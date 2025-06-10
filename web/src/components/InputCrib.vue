<template>
  <div class="flex gap-4 mx-4 my-5">
    <input
      type="number"
      placeholder="x1"
      class="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white"
    />
    <input
      type="number"
      placeholder="y1"
      class="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white"
    />
    <input
      type="number"
      placeholder="x2"
      class="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white"
    />
    <input
      type="number"
      placeholder="y2"
      class="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white"
    />
  </div>
</template>

<script setup>
import { ref } from "vue";
import axios from "axios";

const x1 = ref();
const y1 = ref();
const x2 = ref();
const y2 = ref();

async function getPositionCrib() {
    try {
        const data = await axios.get("/position-crib")
    } catch (error) {
        console.log(error)
    }
}
</script>