<script setup>

</script>

<template>
  <div class="container">
    <!-- <div class="one">
      <img src="../assets/img/images.jpg" alt="" style="height: 60px;width:60px">
      <img src="../assets/img/images.jpg" alt="" style="height: 60px;width:60px">
      <img src="../assets/img/images.jpg" alt="" style="height: 60px;width:60px">
      <img src="../assets/img/images.jpg" alt="" style="height: 60px;width:60px">
      <img src="../assets/img/images.jpg" alt="" style="height: 60px;width:60px">
    </div> -->
    <div class="two" style="text-align: center; height: 30px; font-size: 20px;">
      <span>中国电子科技集团公司第五十四研究所</span>
    </div>
    <div class="two" style="text-align: center; height: 30px; font-size: 20px;">
      <span>北京邮电大学</span>
    </div>
  </div>
</template>

<style scoped>
.container {
  height: 100px;
  background: gray;
  color: white;
  font-size: 20px;
}

.one {
  height: 80px;
  background: grey;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.two {
  height: 40px;
  background: gray;
  font-size: 13px;
  line-height: 40px;

  span {
    margin: 0 200px;
  }
}
</style>
