<template>
  <div class="tab1">
    <div class="one">
      <div class="my-h1" style="width: 100px; height: 33px; margin-left: 5px;">选择信号集:</div>
      <el-select v-model="selectedSignals" placeholder="请选择信号集" style="width:280px">
        <el-option v-for="signal in signals" :key="signal.value" :label="datasetLabel(signal.value)" :value="signal.value"></el-option>
      </el-select>
      <el-button @click="loadSignals" class="button" style="margin-left: 10px" type="success">加载</el-button>

      <!-- 新增部分：文本输入框
      <!-- <el-input v-model="dataPath" placeholder="输入数据路径" style="width: 250px; height: 33px; margin-left: 10px;" /> -->

      <!-- 新增部分：加载用户数据按钮 -->
      <!-- <el-button @click="loadUserData" class="button" style="margin-left: 10px" type="success">加载用户数据</el-button> -->

      <!-- 新增部分：训练模型按钮 -->
      <!-- <el-button @click="trainModel" style="margin-left: 10px" type="success">训练模型</el-button> --> 
      
      <div class="my-h1" style="width: 120px; height: 33px; margin-left: 5px;">选择攻击方法:</div>
      <el-select v-model="selectedMethod" placeholder="请选择攻击方法" style="width:200px;margin-left: 5px">
        <el-option v-for="mt in attackMethods" :key="mt" :label="mt" :value="mt"></el-option>
      </el-select>
      <div class="my-h1" style="width: 120px; height: 33px; margin-left: 5px;">选择模型:</div>
      <el-select v-model="selectedModel" placeholder="请选择模型" style="width:200px;margin-left: 5px">
        <el-option v-for="mt in modelOptions" :key="mt" :label="mt" :value="mt"></el-option>
      </el-select>

    </div>

   

    <div class="two">
      <el-steps :active="activeStepStatus" align-center>
        <el-step title="读取数据"/>
        <el-step title="处理数据"/>
        <!-- <el-step title="增强数据"/> -->
        <el-step title="添加扰动"/>
        <el-step title="识别样本"/>
        <el-step title="输出样本"/>
      </el-steps>
    </div>
    <div class="main">
      <div class="left" style="outline: none">
        <div class="signal-stage">
          <div class="signal-count">信号数量: {{ nodes.length }}{{ datasetSize ? ' / ' + datasetSize : '' }}</div>
          <div class="signal-board" ref="signalBoard">
            <canvas ref="canvas" width="900" height="500"></canvas>
          </div>
          <el-button
            id="actionButton"
            class="action-btn"
            type="success"
            :style="{ width: signalGraphWidth + 'px' }"
            @click="getAttackNodes"
          >
            {{ actionState === 'reset' ? '重置' : '攻击' }}
          </el-button>
        </div>
      </div>

      <div class="right-panel">
        <div class="result-card" v-if="resultsReady">
          <div class="result-title">攻击结果</div>
          <div class="result-line">已处理样本: {{ processedSamples }} / {{ totalSamples }}</div>
          <div class="result-line">攻击样本的成功数量/总数量: {{ successfulAttacks }} / {{ totalSamples }}</div>
          <div class="result-line">攻击成功率: {{ successRateText }}</div>
          <div v-if="featureChangeText" class="result-line">输入特征的变化: {{ featureChangeText }}</div>
          <div v-if="selectedSignalInfo" class="result-line result-sub">{{ selectedSignalInfo }}</div>
        </div>
        <div id="plot" class="right"></div>
        <div id="plotQ" class="right"></div>
        <div id="plotIQ" class="right"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {nextTick, onBeforeUnmount, onMounted, ref} from 'vue'
import Plotly from 'plotly.js-dist-min';
import { subMenuProps } from 'element-plus';
import { api, DEFAULT_ATTACK_METHODS } from '../api'
import { DATASET_SIZES, SIGNAL_RADIUS, boardSize, datasetLabel, layoutNodes, splitIq } from '../signalGraph'


// 获取Canvas元素和上下文
const canvas = ref(null);
const signalBoard = ref(null);
const ctx = ref(null);
const nodes = ref([]); // 存储节点信息
const signals = [
  { value: 'radio1.pkl', label: 'radio1.pkl' },
  { value: 'radio2.pkl', label: 'radio2.pkl' },
  { value: 'radio3.pkl', label: 'radio3.pkl' },
  // { value: 'random_rml2016_subset9.pkl', label: 'random_rml2016_subset9.pkl' },
  // { value: 'random_rml2016_subset7.pkl', label: 'random_rml2016_subset7.pkl' },
  // { value: 'random_rml2016_subset11.pkl', label: 'random_rml2016_subset11.pkl' },
  // { value: 'random_rml2016_subset_100.pkl', label: 'random_rml2016_subset_100.pkl' },
  // { value: 'random_rml2016_subset_200.pkl', label: 'random_rml2016_subset_200.pkl' },
  { value: 'random_rml2016_subset_300.pkl', label: 'random_rml2016_subset_300.pkl' },
  { value: 'random_rml2016_subset_500.pkl', label: 'random_rml2016_subset_500.pkl' },
  { value: 'random_rml2016_subset_1000.pkl', label: 'random_rml2016_subset_1000.pkl' },
  { value: 'random_rml2016_subset_2000.pkl', label: 'random_rml2016_subset_2000.pkl' },
  { value: 'random_rml2016_subset_3000.pkl', label: 'random_rml2016_subset_3000.pkl' },
  { value: 'random_rml2016_subset_5000.pkl', label: 'random_rml2016_subset_5000.pkl' }
  // { value: 'random_rml2016_merged_1.pkl', label: 'random_rml2016_merged_1.pkl' },
  // { value: 'RML2016.10a_dict.pkl', label: 'RML2016.10a_dict.pkl' }
]
const attackMethods = ref([])
const selectedMethod = ref('')
const attackedRml2016Data = ref({})
const activeStepStatus = ref(1)
const isAttacked = ref(false);
var change_ratio = [];
const selectedModel = ref('CNN');
const modelOptions = ref(['RNN', 'CNN']); // 模型选项

//新增部分---------------------------------------------
const processedSamples = ref(0); // 已处理的样本数量
const successfulAttacks = ref(0); // 成功攻击的样本数量
const totalSamples = ref(0); // 总样本数量
const successRateText = ref('');
const featureChangeText = ref('');
const selectedSignalInfo = ref('');
const resultsReady = ref(false);
const signalGraphWidth = ref(900);
const datasetSize = ref(DATASET_SIZES['radio1.pkl'] || 0);
let signalResizeObserver = null;


const selectedNodes = ref(new Set()); // 存储选中的节点索引
const selectedSignals = ref('radio1.pkl');
const actionState = ref('disturb'); // 当前按钮状态 ('disturb'、'defend' 或 'reset')
const colors = {
  'QPSK': '#FF5733',
  'BPSK': '#33FF57',
  'QAM16': '#3357FF',
  'QAM64': '#8C27A1',
  'GFSK': '#A133FF',
  'CPFSK': '#33FFF5',
  'AM-DSB': '#F5FF33',
  'AM-SSB': '#FF8C33',
  'WBFM': '#8CFF33',
  '8PSK': '#338CFF',
  'PAM4': '#FF338C'
}; // 预定义的11种颜色
// ['']


const rml2016Data = ref([]);


const syncSignalGraphWidth = () => {
  const board = signalBoard.value || canvas.value
  if (!board) return
  const width = Math.round(board.clientWidth || board.getBoundingClientRect().width || 900)
  if (width > 0) {
    signalGraphWidth.value = width
  }
}

const toGraphItems = (data) => data.map((item) => {
  const iq = splitIq(item.iq_data)
  return {
    color: colors[item.type] || item.color || '#000',
    iq_data: item.iq_data,
    I_data: iq.I_data || item.I_data,
    Q_data: iq.Q_data || item.Q_data,
    type: item.type,
  }
})

const loadSignals = () => {
  if (selectedSignals.value) {
    datasetSize.value = DATASET_SIZES[selectedSignals.value] || datasetSize.value
    api.get('/load_data', { params: { file: selectedSignals.value } })
        .then(response => {
          if (response.data.code === '00000') {
            const data = response.data.data;
            const total = Number(response.data.total) || data.length
            datasetSize.value = total
            DATASET_SIZES[selectedSignals.value] = total
            const graphItems = toGraphItems(data)
            nodes.value = layoutNodes(graphItems)
            totalSamples.value = total
            rml2016Data.value = graphItems
            isAttacked.value = false
            resultsReady.value = false
            drawNodes();
            nextTick(syncSignalGraphWidth);
          } else {
            console.error('Error loading signal:', response.data.message)
          }
        })
        .catch(error => {
          console.error('Error loading signal:', error)
        })
  } else {
    console.error('No signal selected')
  }
}

onMounted(() => {
  ctx.value = canvas.value.getContext('2d');

  // 处理节点点击事件，选择或取消选择节点，并展示波形图
  canvas.value.addEventListener('click', (event) => {
    const {offsetX, offsetY} = event;
    nodes.value.forEach((node, index) => {
      const hit = SIGNAL_RADIUS + 8
      if (Math.hypot(node.x - offsetX, node.y - offsetY) < hit) {
        if (selectedNodes.value.has(index)) {
          selectedNodes.value.delete(index);
        } else {
          selectedNodes.value.add(index);
        }
        drawNodes();
        showIPlot(index); // 展示波形图
        showQPlot(index); // 展示波形图
        showIQPlot(index);

        if (resultsReady.value) {
          selectedSignalInfo.value = "当前信号编号: " + index + ",    输入特征的变化: " + Number(change_ratio[index] || 0).toFixed(4);
        } 
      }
    });
  });


  // axios.get('http://10.161.41.6:8080/getAll2016FileNames')
  /*axios.get('http://127.0.0.1:8080/getAll2016FileNames')
      .then(response => {
        console.log('Signals:', response.data)
        if (response.data.code === '00000') {
          signals.value = response.data.data
        } else {
          console.error('Error fetching signals:', response.data.message)
        }
      })
      .catch(error => {
        console.error('Error fetching signals:', error)
      })*/

  // 初始化攻击方法
  // axios.get('http://10.161.41.6:8080/getAllAttackMethods')
  attackMethods.value = DEFAULT_ATTACK_METHODS
  selectedMethod.value = selectedMethod.value || DEFAULT_ATTACK_METHODS[0]
  api.get('/getAllAttackMethods')
      .then(response => {
        if (response.data.code === '00000') {
          attackMethods.value = response.data.data
          selectedMethod.value = selectedMethod.value || attackMethods.value[0]
        } else {
          console.error('Error fetching attackMethods:', response.data.message)
        }
      })
      .catch(error => {
        console.error('Error fetching attackMethods:', error)
        attackMethods.value = DEFAULT_ATTACK_METHODS
      })

  // 初始化显示初始状态
  drawInitialState();
  nextTick(syncSignalGraphWidth);
  if (typeof ResizeObserver !== 'undefined' && (signalBoard.value || canvas.value)) {
    signalResizeObserver = new ResizeObserver(() => {
      syncSignalGraphWidth()
    })
    signalResizeObserver.observe(signalBoard.value || canvas.value)
    window.addEventListener('resize', syncSignalGraphWidth)
  }
});

onBeforeUnmount(() => {
  if (signalResizeObserver) {
    signalResizeObserver.disconnect()
    signalResizeObserver = null
  }
  window.removeEventListener('resize', syncSignalGraphWidth)
});

// 未加载前不放占位节点，避免把 8 个示例点当成数据集
function initializeNodes() {
  nodes.value = [];
}

// 自定义函数绘制圆角矩形
function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.arcTo(x + width, y, x + width, y + radius, radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.arcTo(x + width, y + height, x + width - radius, y + height, radius);
  ctx.lineTo(x + radius, y + height);
  ctx.arcTo(x, y + height, x, y + height - radius, radius);
  ctx.lineTo(x, y + radius);
  ctx.arcTo(x, y, x + radius, y, radius);
  ctx.closePath();
}


function drawNodes() {
  const size = boardSize(nodes.value.length)
  canvas.value.width = size.width
  canvas.value.height = size.height
  nextTick(syncSignalGraphWidth)

  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height);
  ctx.value.fillStyle = '#c2e59c';
  drawRoundedRect(ctx.value, 0, 0, canvas.value.width, canvas.value.height, 40);
  ctx.value.fill();  // 填充矩形
  ctx.value.setLineDash([20, 15]);
  ctx.value.strokeStyle = 'grey';
  ctx.value.lineWidth = 4;
  ctx.value.stroke();
  ctx.value.font = '12px Arial';

  if (!nodes.value.length) {
    ctx.value.setLineDash([]);
    ctx.value.fillStyle = '#2e7d32';
    ctx.value.font = '16px Arial';
    ctx.value.fillText('请先加载所选数据集', 340, 250);
    return
  }

  nodes.value.forEach((node, index) => {
    const originalNode = rml2016Data.value[index];  // 获取原始节点数据
    const radius = SIGNAL_RADIUS
    // console.log("index:", index)
    // console.log("originalNode color:", colors[originalNode.type])
    // console.log("originalNode type:", originalNode.type)
    // console.log("Original color:", originalNode.color);  // 输出原始节点的颜色
    // console.log("Current node color:", node.color); 

    // 如果尚未攻击，直接显示原始节点颜色，绘制完整圆
    if (!isAttacked.value) {
      ctx.value.fillStyle = node.color;  // 设置节点颜色
      ctx.value.beginPath();
      ctx.value.arc(node.x, node.y, radius, 0, Math.PI * 2);  // 绘制完整圆
      ctx.value.fill();
    } else {
      // 攻击后处理类型变化与否的逻辑
      const hasChanged = originalNode && node.type !== originalNode.type;

        

      if (hasChanged) {
        // 对类型发生变化的节点进行颜色分割

        // 左半部分原颜色
        ctx.value.fillStyle = colors[originalNode.type];
        ctx.value.beginPath();
        ctx.value.arc(node.x, node.y, radius, Math.PI * 1.5, Math.PI * 0.5);
        ctx.value.fill();

        // 右半部分新颜色
        ctx.value.fillStyle = node.color;
        ctx.value.beginPath();
        ctx.value.arc(node.x, node.y, radius, Math.PI * 0.5, Math.PI * 1.5);
        ctx.value.fill();
        // console.log("index:", index)
        // console.log("originalNode color:", colors[originalNode.type])
        // console.log("originalNode type:", originalNode.type)
        // console.log("node color:", colors[node.type])
        // console.log("node color:", node.color)
        // console.log("node type:", node.type)
      } else {
        // 对未发生变化的节点绘制完整圆
        ctx.value.fillStyle = node.color;
        ctx.value.beginPath();
        ctx.value.arc(node.x, node.y, radius, 0, Math.PI * 2);  // 依然绘制完整圆
        ctx.value.fill();

        // 绘制矩形框
        ctx.value.strokeStyle = 'red';
        ctx.value.lineWidth = 3;
        ctx.value.strokeRect(node.x - radius - 4, node.y - radius - 4, (radius + 4) * 2, (radius + 4) * 2);
      }
    }

    ctx.value.fillStyle = 'black';
    ctx.value.fillText(`信号${index}`, node.x - 20, node.y + radius + 14);

    // 如果节点被选中，则显示边框
    if (selectedNodes.value.has(index)) {
      ctx.value.strokeStyle = 'yellow';
      ctx.value.lineWidth = 3;
      ctx.value.stroke();
    }
  });
}

// 绘制标题，根据当前状态显示相应文字
function drawTitle() {
  ctx.value.clearRect(300, 280, 500, 100); // 扩大清除范围，确保删除所有文字
  ctx.value.fillStyle = 'black';
  if (actionState.value === 'disturb') {
    ctx.value.fillText('原始信号类别', 350, 350);
  } else if (actionState.value === 'defend') {
    ctx.value.fillText('添加扰动之后信号类别', 350, 350);
  } else if (actionState.value === 'reset') {
    ctx.value.fillText('防御之后信号类别', 350, 350);
  }
}



// 展示I信号的波形图
function showIPlot(index) {
  console.log(nodes);
  const data1 = nodes.value[index];
  const data2 = rml2016Data.value[index];
  console.log(data2)

  if (data1) {
    const { I_data: I_data1, Q_data: Q_data1, type: type1 } = data1;
    console.log(I_data1);

    // 计算第一个数据集的幅值
    // const amplitude1 = I_data1.map((I, index) => Math.sqrt(I ** 1 + 0 * Q_data1[index] ** 2));
    const amplitude1 = I_data1.map((I, index) => I ** 1 + 0 * Q_data1[index] ** 2);


    let trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `原始样本类型 ${type1} Signal`,
      line: { color: 'blue' }
    };

    // 设置默认 layout
    const layout = {
      title: `${index} - ${type1} 的I路信号`,
      xaxis: { title: attackedRml2016Data.value.change_ratio || '符号索引' },
      yaxis: { title: '幅度' }
    };

    // 如果 actionState.value 是 'disturb'，只画 trace1，否则画两条曲线
    if (actionState.value === 'disturb') {
      Plotly.newPlot('plot', [trace1], layout);
    } else if (data2) {
      console.log(data2);
      const { I_data: I_data2, Q_data: Q_data2, type: type2 } = data2;
      console.log(I_data2)
      const amplitude2 = I_data2.map((I, index) => I ** 1 + 0 * Q_data2[index] ** 2);
      const trace2 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude2, // 纵坐标为幅值
        mode: 'lines',
        name: `原始样本类型 ${type2} Signal`,
        line: { color: 'blue' }
      };

      trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `扰动样本类型 ${type1} Signal`,
      line: { color: 'red' }
    };

      const amplitude3 = [];
      for(var i = 0; i<I_data2.length; i++){
        amplitude3[i] = amplitude1[i] - amplitude2[i]
      }
      // console.log(`"amp1: ${amplitude1}, amp2: ${amplitude2}, amp3: ${amplitude3}`)
      const trace3 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude3, // 纵坐标为幅值
        mode: 'lines',
        name: `变化幅值`,
        line: { color: 'green' }
      };

      // 绘制两条曲线
      Plotly.newPlot('plot', [trace2, trace1, trace3], layout);
    }
  }
}

function showQPlot(index) {
  // console.log(nodes);
  const data1 = nodes.value[index];
  const data2 = rml2016Data.value[index];
  // console.log(data2)

  if (data1) {
    const { I_data: I_data1, Q_data: Q_data1, type: type1 } = data1;
    // console.log(I_data1);

    // 计算第一个数据集的幅值
    // const amplitude1 = I_data1.map((I, index) => Math.sqrt(I ** 1 + 0 * Q_data1[index] ** 2));
    const amplitude1 = I_data1.map((I, index) => Q_data1[index] ** 1);


    let trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `原始样本类型 ${type1} Signal`,
      line: { color: 'blue' }
    };

    // 设置默认 layout
    const layout = {
      title: `${index} - ${type1} 的Q路信号`,
      xaxis: { title: attackedRml2016Data.value.change_ratio || '符号索引' },
      yaxis: { title: '幅度' }
    };

    // 如果 actionState.value 是 'disturb'，只画 trace1，否则画两条曲线
    if (actionState.value === 'disturb') {
      Plotly.newPlot('plotQ', [trace1], layout);
    } else if (data2) {
      // console.log(data2);
      const { I_data: I_data2, Q_data: Q_data2, type: type2 } = data2;
      // console.log(I_data2)
      const amplitude2 = I_data2.map((I, index) =>Q_data2[index] ** 1);
      const trace2 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude2, // 纵坐标为幅值
        mode: 'lines',
        name: `原始样本类型 ${type2} Signal`,
        line: { color: 'blue' }
      };

      trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `扰动样本类型 ${type1} Signal`,
      line: { color: 'red' }
    };

      const amplitude3 = [];
      for(var i = 0; i<I_data2.length; i++){
        amplitude3[i] = amplitude1[i] - amplitude2[i]
      }
      // console.log(`"amp1: ${amplitude1}, amp2: ${amplitude2}, amp3: ${amplitude3}`)
      const trace3 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude3, // 纵坐标为幅值
        mode: 'lines',
        name: `变化幅值`,
        line: { color: 'green' }
      };

      // 绘制两条曲线
      Plotly.newPlot('plotQ', [trace2, trace1, trace3], layout);
    }
  }
}

// function calculate(index) {
//       const data1 = nodes.value[index];
//       const data2 = rml2016Data.value[index];
//       const { I_data: I_data1, Q_data: Q_data1, type: type1 } = data1;
//       const amplitude1 = I_data1.map((I, index) => Math.sqrt(I_data1[index] ** 2 + Q_data1[index] ** 2));
//       const { I_data: I_data2, Q_data: Q_data2, type: type2 } = data2;
//       const amplitude2 = I_data2.map((I, index) =>Math.sqrt(I_data2[index] ** 2 + Q_data2[index] ** 2));
//       const pertubRatios = [];
//       const amplitude3 = [];
//       let pertubSum = 0;
//       let amplitude1Max = 0
//       for(var i = 0; i < amplitude1.length; i++) {
//         if(amplitude1[i] >= amplitude1Max) {
//           amplitude1Max = amplitude1[i]
//         }
//       }      
//       let suma3 = 0
//       let suma4 = 0 
//       for(var i = 0; i<I_data2.length; i++){
//         amplitude3[i] = amplitude1[i] - amplitude2[i]
//         suma3 += Math.abs(amplitude3[i]) ** 2
//         suma4 += Math.abs(amplitude1[i]) ** 2
//         // console.log(`pertubRatios${i}: ${pertubRatios[i] * 100}`)
//       }

//       let pertubRatio = suma3 / suma4
//       change_ratio.push(pertubRatio)
//       console.log(`pertubRatio: ${pertubRatio * 100}`)
// }

function calculate(index) {
  if (Number.isFinite(Number(change_ratio[index]))) {
    return Number(change_ratio[index])
  }
  const data1 = nodes.value[index];
  const data2 = rml2016Data.value[index];
  if (!data1 || !data2 || !data1.I_data || !data2.I_data) {
    return 0
  }
  const { I_data: I_data1, Q_data: Q_data1 } = data1;
  const { I_data: I_data2, Q_data: Q_data2 } = data2;
  let perturbEnergy = 0
  let originalEnergy = 0
  const len = Math.min(I_data1.length, I_data2.length)
  for (let i = 0; i < len; i++) {
    const dI = Number(I_data1[i] || 0) - Number(I_data2[i] || 0)
    const dQ = Number(Q_data1[i] || 0) - Number(Q_data2[i] || 0)
    const oI = Number(I_data2[i] || 0)
    const oQ = Number(Q_data2[i] || 0)
    perturbEnergy += dI * dI + dQ * dQ
    originalEnergy += oI * oI + oQ * oQ
  }
  return originalEnergy > 0 ? perturbEnergy / originalEnergy : 0
}


function showIQPlot(index) {
  // console.log(nodes);
  const data1 = nodes.value[index];
  const data2 = rml2016Data.value[index];
  // console.log(data2)

  if (data1) {
    const { I_data: I_data1, Q_data: Q_data1, type: type1 } = data1;
    // console.log(I_data1);

    // 计算第一个数据集的幅值
    const amplitude1 = I_data1.map((I, index) => Math.sqrt(I_data1[index] ** 2 + Q_data1[index] ** 2));
    // const amplitude1 = I_data1.map((I, index) => Q_data1[index] ** 1);


    let trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `原始样本类型 ${type1} Signal`,
      line: { color: 'blue' }
    };

    // 设置默认 layout
    const layout = {
      title: `${index} - ${type1} 的包络`,
      xaxis: { title: attackedRml2016Data.value.change_ratio || '符号索引' },
      yaxis: { title: '幅度' }
    };

    // 如果 actionState.value 是 'disturb'，只画 trace1，否则画两条曲线
    if (actionState.value === 'disturb') {
      Plotly.newPlot('plotIQ', [trace1], layout);
    } else if (data2) {
      // console.log(data2);
      const { I_data: I_data2, Q_data: Q_data2, type: type2 } = data2;
      // console.log(I_data2)
      const amplitude2 = I_data2.map((I, index) =>Math.sqrt(I_data2[index] ** 2 + Q_data2[index] ** 2));
      const trace2 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude2, // 纵坐标为幅值
        mode: 'lines',
        name: `原始样本类型 ${type2} Signal`,
        line: { color: 'blue' }
      };



      trace1 = {
      x: Array.from(Array(I_data1.length).keys()), // 横坐标为符号索引
      y: amplitude1, // 纵坐标为幅值
      mode: 'lines',
      name: `扰动样本类型 ${type1} Signal`,
      line: { color: 'red' }
    };

      // const pertubRatios = [];
      const amplitude3 = [];
      // let pertubSum = 0;
      // let amplitude1Max = 0
      // for(var i = 0; i < amplitude1.length; i++) {
      //   if(amplitude1[i] >= amplitude1Max) {
      //     amplitude1Max = amplitude1[i]
      //   }
      // }
      // amplitude1Max = amplitude1Max / 4  
      // // console.log(Math.max(amplitude1))
      // let cnt = 0
      for(var i = 0; i<I_data2.length; i++){
        amplitude3[i] = amplitude1[i] - amplitude2[i]
      //   if(Math.abs(amplitude1[i]) >= amplitude1Max) {
      //     // ((Math.abs(amplitude2[i]) ** 2 + Math.abs(amplitude1[i]) ** 2) / 2)
      //     pertubRatios[cnt] = Math.abs(amplitude3[i]) ** 2 / (Math.abs(amplitude1[i]) ** 2)
      //     pertubSum += pertubRatios[cnt]
      //     cnt++
        }
      //   // console.log(`pertubRatios${i}: ${pertubRatios[i] * 100}`)
      // }

      // let pertubRatio = pertubSum / pertubRatios.length
      // console.log(`pertubRatio: ${pertubRatio * 100}`)

      calculate(index);

      // console.log(`"amp1: ${amplitude1}, amp2: ${amplitude2}, amp3: ${amplitude3}`)
      const trace3 = {
        x: Array.from(Array(I_data2.length).keys()), // 横坐标为符号索引
        y: amplitude3, // 纵坐标为幅值
        mode: 'lines',
        name: `变化幅值`,
        line: { color: 'green' }
      };

      // 绘制两条曲线
      Plotly.newPlot('plotIQ', [trace2, trace1, trace3], layout);
    }
  }
}

// 绘制初始状态的信号图
function drawInitialState() {
  initializeNodes();
  drawNodes();
}


// 防御选中的节点
function defendNodes() {
  selectedNodes.value.forEach((index) => {
    // 恢复节点的原始颜色
    nodes.value[index].color = nodes.value[index].originalColor;
  });
  drawNodes();
  actionState.value = 'reset'; // 设置按钮状态为“重置”
  selectedNodes.value.clear(); // 清空选中状态
  // drawTitle(); // 更新标题为当前状态
}

// 重置为初始状态
function resetState() {
  attackedRml2016Data.value = {}
  activeStepStatus.value = 1
  actionState.value = 'disturb'; // 重置按钮状态为“扰动”
  successRateText.value = ''
  featureChangeText.value = ''
  selectedSignalInfo.value = ''
  resultsReady.value = false
  nodes.value = layoutNodes(toGraphItems(rml2016Data.value));
  isAttacked.value = false;
  
  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height); // 清空canvas
  document.getElementById('plot').innerHTML = ''; // 清空波形图显示区域
  document.getElementById('plotQ').innerHTML = ''; // 清空波形图显示区域
  document.getElementById('plotIQ').innerHTML = ''; // 清空波形图显示区域

  drawNodes();
  // drawTitle(); // 更新标题为当前状态

  // 重置样本处理状态数据------------------------------------
  change_ratio = []
  processedSamples.value = 0;
  successfulAttacks.value = 0;
  totalSamples.value = rml2016Data.value.length || 0;
  successRateText.value = ''
  featureChangeText.value = ''
  selectedSignalInfo.value = ''
}

const getAttackNodes = async () => {
  if (actionState.value === 'reset') {
    resetState();
  } else {
    new Promise((resolve) => {
      let timer = 200 + 200 * Math.random()
      for (let index = 2; index <= 6; index++) {
        setTimeout(() => {
          activeStepStatus.value = index
          if (index === 6) resolve()
        }, index * timer)
      }
    }).then(async ()=>{
      console.log('model' + selectedModel.value)
      const res = await api.get('/attack', {
        params: {
          attack_name: selectedMethod.value,
          dataset_name: `${selectedSignals.value}`,
          model_name: `${selectedModel.value == 'RNN' ? 'Based_LSTM' : 'VTCNN2'}`
        }
      })

      if (res.data.code === '00000') {
        const data = res.data.attackedRml2016Data
        const sucInfo = res.data.info.successInfo
        const transformInfo = res.data.info.transformInfo
        attackedRml2016Data.value = data
        nodes.value = layoutNodes(toGraphItems(data))

        isAttacked.value = true;
        change_ratio = data.map(item => Number(item.change_ratio) || 0)
        const mean_ = change_ratio.length
          ? change_ratio.reduce((sum, value) => sum + value, 0) / change_ratio.length
          : Number(transformInfo) || 0
        const rate = Number(sucInfo)
        const totalCount = Number(res.data.info.totalCount) || data.length
        let successCount = Number(res.data.info.successCount)
        if (!Number.isFinite(successCount)) {
          successCount = Number.isFinite(rate) ? Math.round(rate * totalCount) : 0
        }
        totalSamples.value = totalCount
        processedSamples.value = totalCount
        successfulAttacks.value = successCount
        successRateText.value = Number.isFinite(rate) ? (rate * 100).toFixed(2) + '%' : String(sucInfo)
        featureChangeText.value = Number(mean_ || 0).toFixed(4)
        selectedSignalInfo.value = ''
        resultsReady.value = true
        actionState.value = 'reset'
        drawNodes()
      } else {
        console.error('Error fetching attackedRml2016Data:', res.data.message)
      }
    })

  }

}

// // 新增部分：存储用户输入的数据路径
// const dataPath = ref('');

// // 加载用户输入的路径数据
// const loadUserData = () => {
//   if (dataPath.value) {
//     axios
//       .get(`http://10.161.41.3:5000/load_data?file=${dataPath.value}`)
//       .then((response) => {
//         console.log('User data:', response.data);
//         if (response.data.code === '00000') {
//           const data = response.data.data;
//           nodes.value = data.map((item, index) => ({
//             x: ((index % 18) + 1) * 45,
//             y: (Math.floor(index / 18) + 0.5) * 70,
//             color: colors[item.type] || '#000', // 根据type分配颜色
//             iq_data: item.iq_data,
//             I_data: item.iq_data[0],
//             Q_data: item.iq_data[1],
//             type: item.type
//           }));
//           drawNodes();
//         } else {
//           console.error('Error loading user data:', response.data.message);
//         }
//       })
//       .catch((error) => {
//         console.error('Error loading user data:', error);
//       });
//   } else {
//     console.error('No data path provided');
//   }
//   console.log('data path:', dataPath.value);
// };

// // 调用后端模型训练服务
// const trainModel = () => {
//   if (dataPath.value) {
//     axios
//       .post('http://10.161.41.3:5000/train_model', {
//         data_path: dataPath.value
//       })
//       .then((response) => {
//         console.log('Model training response:', response.data);
//         if (response.data.code === '00000') {
//           console.log('Model training started successfully');
//         } else {
//           console.error('Error starting model training:', response.data.message);
//         }
//       })
//       .catch((error) => {
//         console.error('Error training model:', error);
//       });
//   } else {
//     console.error('No data path provided for training');
//   }
// };


</script>

<style scoped>

canvas {
  display: block;
  border: none;
}

.signal-board {
  max-width: 100%;
  max-height: min(64vh, 640px);
  overflow: auto;
  width: fit-content;
}

.signal-stage {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  width: fit-content;
  max-width: 100%;
}

.signal-count {
  margin-bottom: 8px;
  font-weight: bold;
  color: #2e7d32;
}

.action-btn {
  margin-top: 12px;
  box-sizing: border-box;
  min-width: 0 !important;
}

.one {
  padding: 0 20px;
  display: flex;
  justify-content: flex-start;
  height: 60px;
  width: 1000px
}

.two {
  margin-bottom: 15px;
}

.my-h1 {
  width: 130px;
  height: 60px;
  padding-top: 5px;
  font-weight: bold;
}

.main {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
}

.left {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-left: 10px;
  max-width: 100%;
}

.right-panel {
  flex: 1;
  min-width: 420px;
  max-width: 720px;
  margin: 0 20px 0 10px;
}

.result-card {
  margin-bottom: 16px;
  padding: 16px 20px;
  border: 1px solid #c8e6c9;
  background: #f3faf3;
  color: #2e7d32;
  font-weight: bold;
  font-size: 18px;
  line-height: 1.8;
}

.result-title {
  margin-bottom: 6px;
  font-size: 20px;
}

.result-line {
  word-break: break-all;
}

.result-sub {
  margin-top: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #388e3c;
}

.right {
  width: 100%;
  min-height: 280px;
}


</style>


