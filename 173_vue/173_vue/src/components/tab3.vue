<template>
  <div style="display: flex; flex-direction: column; align-items: flex-start;">
    <div class="left">
      <h3>选择网络拓扑</h3>
      <el-select v-model="selectedNetwork" placeholder="请选择网络拓扑">
        <el-option v-for="network in networks" :key="network" :label="network" :value="network"></el-option>
      </el-select>
      <el-button @click="loadNetwork" type="success">加载</el-button>
      <div>
        <el-button id="identify-button" @click="identifyKeyNodes" type="success">关键节点筛选</el-button>
        <!-- <el-button id="attack-button" style="display: none; " type="success"> </el-button> -->
      </div>
    </div>
    <div id="graph-container"></div>
  </div>

  <!-- //新增部分 ------------------------------- -->
  <!-- <div style="margin-bottom: 20px; width: 100%; height: 10px;">
        <el-progress :percentage="currentProgress" status="success" />
        <div style="display: flex; justify-content: space-between; margin-top: 10px;"> -->
          <!-- <span v-if="totalSamples === 0">未加载样本数据</span>
          <span v-else>已处理样本：{{ processedSamples }} / {{ totalSamples }}</span>
          <span v-if="totalSamples !== 0">检测数量/恶意数据总数量：{{ successfulAttacks2 }} / {{ totalSamples/2 }}</span>
          <span v-if="totalSamples !== 0">虚警数量/非恶意数据总数量：{{ totalSamples / 2 - successfulAttacks1 }} / {{ totalSamples / 2 }}</span> -->
        <!-- </div> -->
         <!-- 将 sucInfoText 移动到进度条下方 -->
  <!-- </div> -->
        <!-- <div id="sucInfoText" style="margin: 20px auto; width: 500px; font-weight: bold; font-size: 20px; color: #4CAF50; text-align: center;"> -->
        <!-- </div> -->

</template>
<script setup>
import {onMounted, ref} from 'vue'
import { api } from '../api'

const cy = ref(null)
const networks = ref([])
const selectedNetwork = ref('')
const timeout = 2; // 
let currentProgress = 0;

const loadNetwork = () => {
  if (selectedNetwork.value) {
    // axios.get(`http://10.161.41.6:8080/graph?fileName=${selectedNetwork.value}`)
    api.get('/graph', { params: { fileName: selectedNetwork.value } })
        .then(response => {
          console.log('Graph data:', response.data)
          if (response.data.code === '00000') {
            const graphData = response.data.data
            cy.value.elements().remove()
            cy.value.add([
              ...graphData.nodes.map(node => ({data: {id: node.id}})),
              ...graphData.edges.map(edge => ({data: {source: edge.source, target: edge.target}}))
            ])
            cy.value.layout({name: 'cose'}).run()
          } else {
            console.error('Error loading network:', response.data.message)
          }
        })
        .catch(error => {
          console.error('Error loading network:', error)
        })
  } else {
    console.error('No network selected')
  }
}

const identifyKeyNodes = () => {
  if (selectedNetwork.value) {
    api.post('/getSolution', {
      step_ratio: 0.01,
      model_file_ckpt: 'nrange_30_50_iter_93300.ckpt',
      data_test_name: [selectedNetwork.value]
    })
        .then(response => {
          console.log('Solution data:', response.data)
          if (response.data.code === '00000' && response.data.data && response.data.data.solution) {
            const keyNodes = response.data.data.solution
            keyNodes.forEach(nodeId => {
              console.log(nodeId)
              // simulateProgress();
              // setTimeout(() => {
              //   setTimeout(() => { cy.value.getElementById(nodeId.toString()).style('background-color', 'red')}, 1000);
              // }, 5000)
              cy.value.getElementById(nodeId.toString()).style('background-color', 'red')
            })
            document.getElementById('attack-button').style.display = 'block'
          } else {
            console.error('Error identifying key nodes:', response.data.message)
          }
        })
        .catch(error => {
          console.error('Error identifying key nodes:', error)
        })
  } else {
    console.error('No network selected')
  }
}

//绘制进度条进度条的变化----------------------
// const drawProgress = (processed, total) => {
//   console.log("draw")
//   const canvas = document.getElementById('progressCanvas');
//   if (!canvas) return; // 修复获取 canvas 可能为空的问题
//   const ctx = canvas.getContext('2d');

//   const percentage = (processed / total) * 100;

//   // 清除画布
//   ctx.clearRect(0, 0, canvas.width, canvas.height);

//   // 绘制背景条
//   ctx.fillStyle = '#e0e0e0';
//   ctx.fillRect(0, 0, canvas.width, canvas.height);

//   // 绘制进度条
//   ctx.fillStyle = (percentage < 50) ? '#f76c6c' : '#76c7c0';
//   ctx.fillRect(0, 0, (canvas.width * percentage) / 100, canvas.height);

//   // 绘制进度文本
//   ctx.fillStyle = '#000';
//   ctx.font = '16px Arial';
//   ctx.fillText(`${Math.round(percentage)}%`, canvas.width / 2 - 20, canvas.height / 2 + 5);
// };

// // 模拟进度条的变化----------------------------------
// const simulateProgress = () => {
//   const step = 1;

//   console.log("simulate:")
//   const interval = setInterval(() => {
//     if (currentProgress <= timeout) {
//       currentProgress += step;
//       if (currentProgress > timeout) {
//         currentProgress = timeout;
//       }
//       // processedSamples.value = currentProgress;
//       // progressPercentage.value = (processedSamples.value / totalSamples.value) * 100;
//     console.log("currentProgress:", currentProgress)

//       drawProgress(currentProgress, timeout);
//     } else {
//       clearInterval(interval);
//     }
//   }, 1000); // 每500毫秒更新一次进度
// };

// // 启动时模拟进度条变化--------------------------
// onMounted(() => {
//   drawProgress(currentProgress, timeout);
// });

onMounted(() => {
  cy.value = cytoscape({
    container: document.getElementById('graph-container'),
    elements: [],
    style: [
      {
        selector: 'node',
        style: {
          'background-color': '#0074D9',
          'label': 'data(id)'
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 2,
          'line-color': '#ccc',
        }
      }
    ],
    layout: {
      name: 'cose'
    }
  });
  // Button event handlers


  

  api.get('/getAllFileNames')
      .then(response => {
        console.log('Networks:', response.data)
        if (response.data.code === '00000') {
          networks.value = response.data.data
        } else {
          console.error('Error fetching networks:', response.data.message)
        }
      })
      .catch(error => {
        console.error('Error fetching networks:', error)
      })
})

</script>

<style scoped>

.left {
  display: flex;
  width: 200px;
  flex-direction: column;
  align-items: center;
}

#graph-container {
  position: absolute;
  top: 120px;
  left: 250px;
  width: 80%;
  height: 80%;
  border: 1px solid #ccc;
}

button {
  display: block;
  margin: 10px auto;
  background-color: #67C23A;
}

h3 {
  margin-top: 20px;
}

</style>
