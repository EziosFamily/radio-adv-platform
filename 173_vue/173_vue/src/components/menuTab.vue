<template>
    <div style="display: flex;width:100%">
        <img src="../assets/img/images.jpg" alt="" style="height: 50px;width:50px;margin-right: 50px">
      <el-tabs v-model="activeName" class="demo-tabs" @tab-change="handleClick">
        <template v-for="(tab, index) in menuList">
          <el-tab-pane :label="tab.label" :name="tab.name"></el-tab-pane>
        </template>
      </el-tabs>
    </div>
</template>
<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
const router = useRouter()
const activeName = ref("home");
const menuList = ref([
  {
    name: "home",
    label: "首页",
    url: "/",
  },
  {
    name: "tab1",
    label: "攻击样本成功率测试",
    url: "/tab1",
  },
  {
    name: "tab2",
    label: "恶意数据检测率测试",
    url: "/tab2",
  },
  {
    name: "tab3",
    label: "网络拓扑中的关键节点筛选测试",
    url: "/tab3",
  },
  // {
  //   name: "tab4",
  //   label: "脆弱性评估测试",
  //   url: "/tab4",
  // }
]);

const handleClick = (name) => {
  const url = menuList.value.find((item) => item.name === name).url;
  router.push(url)
};
</script>

<style scoped>
/deep/ .el-tabs__item {
  height: 50px !important;
  width: 33%;
  padding: 0;
  font-size: 20px;
}
/deep/ .el-tabs__nav-scroll .el-tabs__nav{
  width:1500px
}

/deep/ .el-tabs{
  width:70%
}
</style>
