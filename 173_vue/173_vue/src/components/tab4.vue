<template>
  <div class="tab4-container">
    <div class="control-panel">
      <h3>选择神经网络模型</h3>

      <div class="control-group">
        <el-select
            v-model="selectedModel"
            placeholder="请选择模型"
            style="margin-bottom: 20px; width: 300px"
        >
          <el-option
              v-for="model in modelOptions"
              :key="model.value"
              :label="model.label"
              :value="model.value"
          />
        </el-select>
      </div>

      <div class="control-group">
        <el-select
          v-model="selectedSignal"
          placeholder="请选择信号集"
          style="margin-bottom: 20px; width: 300px"
        >
        <el-option
            v-for="signal in signalOptions"
            :key="signal.value"
            :label="signal.label"
            :value="signal.value"
        />
        </el-select>
      </div>

      <div class="control-group">
        <el-select
          v-model="selectedMethod"
          placeholder="请选择攻击方法"
          style="margin-bottom: 20px; width: 300px"
        >
        <el-option
            v-for="attack_methond in methodOptions"
            :key="attack_methond.value"
            :label="attack_methond.label"
            :value="attack_methond.value"
        />
        </el-select>
      </div>

      <div class="control-group">
        <div class="epoch-inputs">
          <div class="input-item">
            <label>迭代次数下限：</label>
            <el-input-number
                v-model="epochInf"
                :min="1"
                :max="100"
                controls-position="right"
                style="width: 120px"
            />
          </div>

          <div class="input-item">
            <label>迭代次数上限：</label>
            <el-input-number
                v-model="epochSup"
                :min="epochInf + 1"
                :max="100"
                controls-position="right"
                style="width: 120px"
            />
          </div>

          <div class="input-item">
            <label>分析次数：</label>
            <el-input-number
                v-model="epoch_analyze"
                :min="1"
                :max="100"
                controls-position="right"
                style="width: 120px"
            />
          </div>
        </div>
      </div>

      <el-button
          type="success"
          @click="handleModelAction"
          :loading="loading"
          style="margin-top: 15px"
      >
        执行分析
      </el-button>
    </div>

    <!-- 图表容器 -->
    <div class="chart-container">
      <div id="attackAnalysisChart" class="plotly-chart"></div>
    </div>

    <!-- 加载提示 -->
    <div v-if="!chartData.raw.length" class="placeholder-text">
      {{ selectedModel ? "点击按钮开始分析" : "请先选择模型类型" }}
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import { api } from '../api'
import Plotly, { validate } from 'plotly.js-dist-min'
import regression from 'regression'

// 响应式数据
const selectedModel = ref('')
const modelOptions = [
  { value: 'CNN', label: '卷积神经网络（CNN）' },
  { value: 'RNN', label: '循环神经网络（RNN）' }
]

const selectedSignal = ref('')
const signalOptions = [
  { value: 'radio1.pkl', label: 'radio1.pkl' },
  { value: 'radio2.pkl', label: 'radio2.pkl' },
  { value: 'radio3.pkl', label: 'radio3.pkl' },
  { value: 'random_rml2016_subset9.pkl', label: 'random_rml2016_subset9.pkl' },
  { value: 'random_rml2016_subset_100.pkl', label: 'random_rml2016_subset_100.pkl' },
  { value: 'random_rml2016_subset_200.pkl', label: 'random_rml2016_subset_200.pkl' },
  { value: 'random_rml2016_subset_500.pkl', label: 'random_rml2016_subset_500.pkl' },
  { value: 'random_rml2016_subset_1000.pkl', label: 'random_rml2016_subset_1000.pkl' },
  { value: 'random_rml2016_subset_2000.pkl', label: 'random_rml2016_subset_2000.pkl' },
  { value: 'random_rml2016_subset_2000.pkl', label: 'random_rml2016_subset_3000.pkl' },
  { value: 'random_rml2016_subset_2000.pkl', label: 'random_rml2016_subset_5000.pkl' }
]

const selectedMethod=ref('')
const methodOptions=[
  {value:'pgd',label:'pgd'},
  {value:'tsmifgsm',label:'tsmifgsm'},
  {value:'mpdsm',label:'mpdsm'}
]

// 新增：迭代次数参数
const epochInf = ref(10)
const epochSup = ref(20)
const epoch_analyze=ref(10)

const chartData = ref({ raw: [], fit: [] })
const loading = ref(false)
const nodes = ref([])
var change_ratio = []
const rawData = ref([])
const totalSamples = ref(100)

const rml2016Data = ref([])

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
}

// 确保上限大于下限
watch(epochInf, (newVal) => {
  if (epochSup.value <= newVal) {
    epochSup.value = newVal + 1
  }
})

watch(epochSup, (newVal) => {
  if (newVal <= epochInf.value) {
    epochInf.value = newVal - 1
  }
})

function calculate(index) {
  const data1 = nodes.value[index]
  const data2 = rml2016Data.value[index]
  if (!data1 || !data2) {
    console.warn(`索引 ${index} 的数据不存在`)
    return 0
  }

  const { I_data: I_data1, Q_data: Q_data1 } = data1
  const amplitude1 = I_data1.map((I, i) => Math.sqrt(I ** 2 + Q_data1[i] ** 2))

  const { I_data: I_data2, Q_data: Q_data2 } = data2
  const amplitude2 = I_data2.map((I, i) => Math.sqrt(I ** 2 + Q_data2[i] ** 2))

  const pertubRatios = []
  let amplitude1Max = Math.max(...amplitude1) / Math.sqrt(2)
  let cnt = 0
  let pertubSum = 0

  for (let i = 0; i < amplitude1.length; i++) {
    if (Math.abs(amplitude1[i]) >= amplitude1Max) {
      const diff = Math.abs(amplitude1[i] - amplitude2[i])
      const ratio = diff ** 2 / (amplitude1[i] ** 2)
      pertubRatios.push(ratio)
      pertubSum += ratio
      cnt++
    }
  }

  return cnt > 0 ? pertubSum / cnt : 0
}

// 执行分析
const handleModelAction = async () => {
  try {
    loading.value = true
    chartData.value = { raw: [], fit: [] }
    rawData.value = []

    if (!selectedModel.value) {
      console.warn('未选择模型')
      return
    }

    // 加载RML2016数据集
    await api.get('/load_data', { params: { file: selectedSignal.value, max_samples: 64 } })
        .then(response => {
          if (response.data.code === '00000') {
            const data = response.data.data
            totalSamples.value = data.length
            rml2016Data.value = data.map(item => ({
              color: colors[item.type] || '#000',
              iq_data: item.iq_data,
              I_data: item.iq_data[0],
              Q_data: item.iq_data[1],
              type: item.type,
            }))
          } else {
            console.error('Error loading signal:', response.data.message)
          }
        })
        .catch(error => {
          console.error('Error loading signal:', error)
        })

    for (let loop = 0; loop < epoch_analyze.value; loop++) {
      console.log(`正在执行第 ${loop + 1} 次分析...`)

      const res = await api.get('/vulnerability', {
        params: {
          model_name: selectedModel.value === 'RNN' ? 'Based_LSTM' : 'VTCNN2',
          attack_name: selectedMethod.value,
          batch_mode: true,
          signal_name: selectedSignal.value,
          dataset_name: selectedSignal.value,
          epoch_inf: epochInf.value,
          epoch_sup: epochSup.value,
          max_samples: 32
        }
      })

      if (res.data.code === '00000') {
        const data = res.data.attackedRml2016Data
        const sucInfo = res.data.info.successInfo

        nodes.value = data.map((item, index) => ({
          x: ((index % 18) + 1) * 45,
          y: (Math.floor(index / 18) + 0.5) * 70,
          I_data: item.iq_data[0],
          Q_data: item.iq_data[1],
          type: item.type,
        }))

        let mean_ = 0
        change_ratio = []
        for (let i = 0; i < nodes.value.length; i++) {
          const ratio = calculate(i)
          change_ratio.push(ratio)
          mean_ += ratio
        }
        mean_ /= nodes.value.length

        if (mean_ >= 0 && sucInfo >= 0) {
          rawData.value.push([mean_, sucInfo])
          console.log(`第 ${loop + 1} 次分析结果:`, [mean_, sucInfo])
        } else {
          console.warn(`第 ${loop + 1} 次数据无效，跳过`)
        }
      } else {
        console.error(`第 ${loop + 1} 次请求失败:`, res.data.message)
      }
    }

    if (rawData.value.length > 1) {
      const result = regression.polynomial(rawData.value, {
        order: 4,
        precision: 6
      })

      let fitPoints = []
      const minX = Math.min(...rawData.value.map(p => p[0]))
      const maxX = Math.max(...rawData.value.map(p => p[0]))
      const step = (maxX - minX) / 500

      for (let x = minX; x <= maxX; x += step) {
        const y = Math.min(result.predict(x)[1], 1)
        fitPoints.push({ x, y })
      }

      // 添加移动平均平滑
      fitPoints = movingAverage(fitPoints, 7); // 窗口大小为7

      // 确保曲线端点不超过100%
      if (fitPoints.length > 0) {
        fitPoints[0].y = Math.min(fitPoints[0].y, 1)
        fitPoints[fitPoints.length - 1].y = Math.min(fitPoints[fitPoints.length - 1].y, 1)
      }

      // +++ 新增：强制曲线单调递增 +++
      let lastY = 0;
      fitPoints = fitPoints.map(point => {
        // 确保y值不小于0
        let y = Math.max(point.y, 0);

        // 确保曲线单调递增
        if (y < lastY) {
          y = lastY; // 如果当前点小于前一点，则取前一点的值
        } else {
          lastY = y; // 更新最后的值
        }

        // 确保y值不超过1
        y = Math.min(y, 1);

        return { x: point.x, y };
      });

      chartData.value = {
        raw: [...rawData.value],
        fit: fitPoints
      }
    } else {
      console.warn('有效数据点不足，无法生成拟合曲线')
      chartData.value = {
        raw: [...rawData.value],
        fit: []
      }
    }

    await nextTick()
    drawAttackAnalysisChart()
  } catch (error) {
    console.error('请求失败:', error)
  } finally {
    loading.value = false
  }
}

// 绘制图表
const drawAttackAnalysisChart = () => {
  Plotly.purge('attackAnalysisChart')

  if (!chartData.value.raw || chartData.value.raw.length === 0) {
    console.warn('无有效数据可绘制')
    return
  }

  const traces = []

  traces.push({
    x: chartData.value.raw.map(p => p[0]),
    y: chartData.value.raw.map(p => p[1]),
    mode: 'markers',
    name: '原始数据',
    marker: {
      size: 8,
      color: '#409EFF',
      opacity: 0.8
    }
  })

  if (chartData.value.fit && chartData.value.fit.length > 0) {
    traces.push({
      x: chartData.value.fit.map(p => p.x),
      y: chartData.value.fit.map(p => p.y),
      mode: 'lines',
      name: '拟合曲线',
      line: {
        color: '#67C23A',
        width: 3,
        shape: 'spline'
      }
    })
  }

  const layout = {
    title: `攻击效果分析 (迭代次数: ${epochInf.value}-${epochSup.value})`,
    xaxis: {
      title: '输入特征变化率',
      tickformat: '.2%',
      gridcolor: '#eee',
      titlefont: { size: 14 }
    },
    yaxis: {
      title: '攻击成功率',
      tickformat: '.1%',
      range: [0, 1],
      gridcolor: '#eee',
      titlefont: { size: 14 }
    },
    plot_bgcolor: '#f9f9f9',
    margin: { t: 40, l: 60, r: 40, b: 60 },
    showlegend: true,
    legend: {
      x: 0.8,
      y: 0.1
    }
  }

  Plotly.newPlot('attackAnalysisChart', traces, layout)
}

// 移动平均平滑函数
function movingAverage(points, windowSize = 3) {
  if (points.length <= windowSize) return points;

  const smoothed = [];
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < points.length; i++) {
    let sum = 0;
    let count = 0;

    for (let j = i - halfWindow; j <= i + halfWindow; j++) {
      if (j >= 0 && j < points.length) {
        sum += points[j].y;
        count++;
      }
    }

    smoothed.push({
      x: points[i].x,
      y: Math.min(sum / count, 1) // 确保平均值不超过100%
    });
  }

  return smoothed;
}
</script>

<style scoped>
.tab4-container {
  padding: 20px;
}

.control-panel {
  display: flex;
  flex-direction: column;
  margin-bottom: 30px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.control-group {
  margin-bottom: 15px;
}

.epoch-inputs {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  align-items: center;
}

.input-item {
  display: flex;
  align-items: center;
}

.input-item label {
  margin-right: 10px;
  font-size: 14px;
  color: #606266;
  min-width: 100px;
}

.chart-container {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 15px;
  background: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.plotly-chart {
  width: 100%;
  height: 600px;
}

.placeholder-text {
  text-align: center;
  color: #909399;
  font-size: 16px;
  margin-top: 30px;
  font-weight: bold;
  padding: 20px;
  border: 1px dashed #ebeef5;
  border-radius: 4px;
  background: #fafafa;
}
</style>