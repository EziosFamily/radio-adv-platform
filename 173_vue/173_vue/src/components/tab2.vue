<template>
  <div class="tab1">
    <div class="one">
      <div class="my-h1" style="width: 100px; height: 33px; margin-left: 5px;">选择信号集:</div>
      <el-select v-model="selectedSignals" placeholder="请选择信号集" style="width:280px">
        <el-option v-for="signal in signals" :key="signal.value" :label="datasetLabel(signal.value)" :value="signal.value"></el-option>
      </el-select>
      <el-button @click="loadSignals" class="button" style="margin-left: 10px" type="success">加载</el-button>
      <div class="my-h1" style="width: 120px; height: 33px; margin-left: 5px;">选择检测方法:</div>
      <el-select v-model="selectedMethod" placeholder="请选择检测方法" style="width:200px;margin-left: 5px">
        <el-option v-for="mt in detectMethods" :key="mt" :label="mt" :value="mt"></el-option>
      </el-select>
      <div class="my-h1" style="width: 120px; height: 33px; margin-left: 5px;">选择模型:</div>
      <el-select v-model="selectedModel" placeholder="请选择模型" style="width:200px;margin-left: 5px">
        <el-option v-for="mt in modelOptions" :key="mt" :label="mt" :value="mt"></el-option>
      </el-select>
      <!-- <div id="sucInfoText" style="margin-left: 10px;width: 500px;"></div> -->
    </div>

    <div v-if="isAttacked" class="legend" style="margin-left: 20px;">
        <div>
          <span v-if="totalSamples !== 0" class="circle" style="background-color: green; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></span>
          <span v-if="totalSamples !== 0" style="margin-left: 10px;">正常信号</span>
        </div>
        <div style="margin-top: 10px;">
          <span v-if="totalSamples !== 0" class="circle" style="background-color: black; width: 20px; height: 20px; border-radius: 50%; display: inline-block;"></span>
          <span v-if="totalSamples !== 0" style="margin-left: 10px;">异常信号</span>
        </div>
      </div>

      <div class="two">
      <el-steps :active="activeStepStatus" align-center>
        <el-step title="读取数据"/>
        <el-step title="处理数据"/>
        <el-step title="增强数据"/>
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
            {{ actionState === 'reset' ? '重置' : '检测' }}
          </el-button>
        </div>
      </div>

      <div class="right-panel">
        <div class="result-card" v-if="resultsReady">
          <div class="result-title">检测结果</div>
          <div class="result-line">检测方法: {{ usedMethod || selectedMethod }}</div>
          <div class="result-line">检测模型: {{ usedModel || selectedModel }}</div>
          <div class="result-line">数据集条数: {{ totalSamples }}</div>
          <div class="result-line">对抗样本检出: {{ detectedMalicious }} / {{ totalSamples }}</div>
          <div class="result-line">干净样本虚警: {{ falseAlarms }} / {{ totalSamples }}</div>
          <div class="result-line">恶意数据检测率: {{ detectionRateText }}</div>
          <div class="result-line">虚警率: {{ falseAlarmRateText }}</div>
          <div v-if="selectedSignalInfo" class="result-line result-sub">{{ selectedSignalInfo }}</div>
        </div>
        <div id="plot" class="right"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import {nextTick, onBeforeUnmount, onMounted, ref} from 'vue'
import Plotly from 'plotly.js-dist-min';
import { api, DEFAULT_DETECT_METHODS } from '../api'
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
  { value: 'random_rml2016_subset_300.pkl', label: 'random_rml2016_subset_300.pkl' },
  { value: 'random_rml2016_subset_500.pkl', label: 'random_rml2016_subset_500.pkl' },
  { value: 'random_rml2016_subset_1000.pkl', label: 'random_rml2016_subset_1000.pkl' },
  { value: 'random_rml2016_subset_2000.pkl', label: 'random_rml2016_subset_2000.pkl' },
  { value: 'random_rml2016_subset_3000.pkl', label: 'random_rml2016_subset_3000.pkl' },
  { value: 'random_rml2016_subset_5000.pkl', label: 'random_rml2016_subset_5000.pkl' }
]
const detectMethods = ref(DEFAULT_DETECT_METHODS)
const selectedMethod = ref(DEFAULT_DETECT_METHODS[0])
const usedMethod = ref('')
const usedModel = ref('')
const attackedRml2016Data = ref({})
const activeStepStatus = ref(1)
const isAttacked = ref(false)
const selectedModel = ref('CNN');
const modelOptions = ref(['RNN', 'CNN']); // 模型选项

//新增部分---------------------------------------------
const processedSamples = ref(0);
const detectedMalicious = ref(0);
const falseAlarms = ref(0);
const maliciousTotal = ref(0);
const cleanTotal = ref(0);
const totalSamples = ref(0);
const detectionRateText = ref('');
const falseAlarmRateText = ref('');
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
  'QAM64': '#FF33A1',
  'GFSK': '#A133FF',
  'CPFSK': '#33FFF5',
  'AM-DSB': '#F5FF33',
  'AM-SSB': '#FF8C33',
  'WBFM': '#8CFF33',
  '8PSK': '#338CFF',
  'PAM4': '#FF338C'
}; // 预定义的11种颜色


const rml2016Data = ref([]);

const syncSignalGraphWidth = () => {
  const board = signalBoard.value || canvas.value
  if (!board) return
  const width = Math.round(board.clientWidth || board.getBoundingClientRect().width || 900)
  if (width > 0) {
    signalGraphWidth.value = width
  }
}

const toGraphItems = (data, colorFn) => data.map((item, index) => {
  const iq = splitIq(item.iq_data)
  return {
    color: colorFn ? colorFn(item, index) : (colors[item.type] || item.color || '#000'),
    iq_data: item.iq_data,
    I_data: iq.I_data || item.I_data,
    Q_data: iq.Q_data || item.Q_data,
    type: item.type,
  }
})

const clearDetectResults = () => {
  processedSamples.value = 0
  detectedMalicious.value = 0
  falseAlarms.value = 0
  maliciousTotal.value = 0
  cleanTotal.value = 0
  detectionRateText.value = ''
  falseAlarmRateText.value = ''
  selectedSignalInfo.value = ''
  resultsReady.value = false
  usedMethod.value = ''
  usedModel.value = ''
}

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
            totalSamples.value = total
            nodes.value = layoutNodes(graphItems)
            rml2016Data.value = graphItems
            clearDetectResults()
            isAttacked.value = false
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
        showIQPlot(index); // 展示波形图
        if (resultsReady.value) {
          selectedSignalInfo.value = '当前信号编号: ' + index + ',    判定: ' + (node.type || '')
        }
      }
    });
  });

   // 按钮点击事件，根据状态执行扰动、防御或重置
   //document.getElementById('actionButton').addEventListener('click', () => {
   //  if (actionState.value === 'disturb') {
   //    disturbSelectedNodes();
   //  } else if (actionState.value === 'defend') {
   //    defendNodes();
   //  } else if (actionState.value === 'reset') {
   //    resetState();
   //  }
   //});

  // axios.get('http://127.0.0.1:8080/getAll2016FileNames')
  //     .then(response => {
  //       console.log('Signals:', response.data)
  //       if (response.data.code === '00000') {
  //         signals.value = response.data.data
  //       } else {
  //         console.error('Error fetching signals:', response.data.message)
  //       }
  //     })
  //     .catch(error => {
  //       console.error('Error fetching signals:', error)
  //     })

  detectMethods.value = DEFAULT_DETECT_METHODS
  selectedMethod.value = selectedMethod.value || DEFAULT_DETECT_METHODS[0]
  api.get('/getAllDetectMethods')
      .then(response => {
        if (response.data.code === '00000' && response.data.data?.length) {
          detectMethods.value = response.data.data
          if (!detectMethods.value.includes(selectedMethod.value)) {
            selectedMethod.value = detectMethods.value[0]
          }
        }
      })
      .catch(error => {
        console.error('Error fetching detectMethods:', error)
        detectMethods.value = DEFAULT_DETECT_METHODS
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

// 绘制节点
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
    const radius = SIGNAL_RADIUS
    ctx.value.fillStyle = node.color;
    ctx.value.beginPath();
    ctx.value.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.value.fill();
    ctx.value.fillStyle = 'black';
    ctx.value.font = '12px Arial'
    ctx.value.fillText(`信号${index}`, node.x - 20, node.y + radius + 14);

    if (isAttacked.value){
      if(node.type == "检测错误") {
        ctx.value.strokeStyle = 'red'; 
        ctx.value.lineWidth = 3; 
        ctx.value.strokeRect(node.x - 15, node.y - 15, 30, 30);
      }
    }

    // 如果节点被选中，则显示边框
    if (selectedNodes.value.has(index)) {
      ctx.value.strokeStyle = 'yellow';
      ctx.value.lineWidth = 3;
      ctx.value.stroke();
    }
  });

  // 根据当前状态绘制标题
  // drawTitle();
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

// 展示IQ信号的波形图
function showIQPlot(index) {
  const data = nodes.value[index];
  if (data) {
    const {I_data, Q_data, type} = data;
    console.log("I_data:", I_data);
    console.log("Q_data:", Q_data);

    const amplitude = I_data.map((I, index) => Math.sqrt(I ** 2 + Q_data[index] ** 2));

    const trace = {
      x: Array.from(Array(I_data.length).keys()), // 横坐标为符号索引
      y: amplitude, // 纵坐标为幅值
      mode: 'lines',
      name: `${type} Signal`,
      line: {color: 'blue'}
    };
    console.log("len", data.type)
    const layout = {
      title: data.type === '检测错误'
        ? (data.color === 'black' ? `信号${index} - 漏检` : `信号${index} - 虚警`)
        : (data.color === 'black' ? `信号${index} - 检出` : `信号${index} - 正常`),         
      //  data.predict_type=='检测错误') ? `信号${index} - 黑错 信号幅度` : `信号${index} - 绿错 信号幅度`,
      xaxis: {title: '符号索引'},
      yaxis: {title: '幅度'}
    };

    Plotly.newPlot('plot', [trace], layout);
  }
}

// 绘制初始状态的信号图
function drawInitialState() {
  initializeNodes();
  drawNodes();
}

// // 扰动选中的节点
// function disturbSelectedNodes() {
//   selectedNodes.value.forEach(index => {
//     if (Math.random() > 0.1) {
//       const randomColor = colors[Math.floor(Math.random() * colors.length)];
//       nodes.value[index].color = randomColor;
//     }
//   });
//   drawNodes();
//   actionState.value = 'reset'; // 更改按钮状态为“防御”
//   document.getElementById('actionButton').textContent = '重置';
//   drawTitle(); // 更新标题为当前状态
// }

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
  activeStepStatus.value = 1
  actionState.value = 'disturb'; // 重置按钮状态为“扰动”
  isAttacked.value = false
  nodes.value = layoutNodes(toGraphItems(rml2016Data.value));
  ctx.value.clearRect(0, 0, canvas.value.width, canvas.value.height); // 清空canvas
  document.getElementById('plot').innerHTML = ''; // 清空波形图显示区域
  clearDetectResults()
  totalSamples.value = rml2016Data.value.length || 0
  drawNodes();
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
    }).then(async () => {
      const res = await api.get('/detect', {
        params: {
          selectedSignals: selectedSignals.value,
          selectedModel: selectedModel.value,
          detect_name: selectedMethod.value
        }
      })
      
      if (res.data.code === '00000') {
        const data = res.data.data
        const clean = data.filter((item) => Number(item.actual_type) === 0)
        const malicious = data.filter((item) => Number(item.actual_type) === 1)
        const n = Number(res.data.dataset_count) || rml2016Data.value.length || clean.length || malicious.length
        datasetSize.value = n
        const source = rml2016Data.value.length === n ? rml2016Data.value : clean
        nodes.value = layoutNodes(source.map((item, index) => {
          const iq = splitIq(item.iq_data)
          const flagged = malicious[index] ? Number(malicious[index].predict_type) === 1 : false
          const falseAlarm = clean[index] ? Number(clean[index].predict_type) === 1 : false
          return {
            color: flagged ? 'black' : 'green',
            iq_data: item.iq_data,
            I_data: iq.I_data || item.I_data,
            Q_data: iq.Q_data || item.Q_data,
            type: (malicious[index] && Number(malicious[index].predict_type) !== 1) || falseAlarm ? '检测错误' : '检测正确',
            actual_type: flagged ? 1 : 0,
            predict_type: flagged || falseAlarm ? 1 : 0,
          }
        }))
        cleanTotal.value = clean.length || n
        maliciousTotal.value = malicious.length || n
        detectedMalicious.value = malicious.filter((item) => Number(item.predict_type) === 1).length
        falseAlarms.value = clean.filter((item) => Number(item.predict_type) === 1).length
        totalSamples.value = n
        processedSamples.value = n
        const rate = maliciousTotal.value
          ? detectedMalicious.value / maliciousTotal.value
          : Number(res.data.accuracy)
        detectionRateText.value = Number.isFinite(rate) ? (rate * 100).toFixed(2) + '%' : String(res.data.accuracy)
        const far = cleanTotal.value ? falseAlarms.value / cleanTotal.value : 0
        falseAlarmRateText.value = Number.isFinite(far) ? (far * 100).toFixed(2) + '%' : ''
        usedMethod.value = res.data.method || selectedMethod.value
        usedModel.value = res.data.model || selectedModel.value
        selectedSignalInfo.value = ''
        resultsReady.value = true
        isAttacked.value = true
        actionState.value = 'reset'
        drawNodes()
      } else {
        console.error('Error fetching attackedRml2016Data:', res.data.message)
      }
    })
  }
}
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
  align-items: center;
  flex-wrap: wrap;
  min-height: 60px;
  width: 100%;
  box-sizing: border-box;
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
