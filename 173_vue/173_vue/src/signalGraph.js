export const DATASET_SIZES = {
  'radio1.pkl': 1046,
  'radio2.pkl': 193,
  'radio3.pkl': 220,
  'random_rml2016_subset_300.pkl': 300,
  'random_rml2016_subset_500.pkl': 500,
  'random_rml2016_subset_1000.pkl': 1000,
  'random_rml2016_subset_2000.pkl': 2000,
  'random_rml2016_subset_3000.pkl': 3000,
  'random_rml2016_subset_5000.pkl': 5000,
}

export const SIGNAL_RADIUS = 10
export const SIGNAL_COL_GAP = 45
export const SIGNAL_ROW_GAP = 70
export const SIGNAL_COLS = 18
export const BOARD_MIN_WIDTH = 900
export const BOARD_MIN_HEIGHT = 500
export const BOARD_PAD = 50

export function datasetLabel(fileName) {
  const n = DATASET_SIZES[fileName]
  return n ? `${fileName}（${n}条）` : fileName
}

export function splitIq(iq) {
  if (!iq) return { I_data: undefined, Q_data: undefined }
  let arr = iq
  while (Array.isArray(arr) && arr.length === 1 && Array.isArray(arr[0])) {
    arr = arr[0]
  }
  if (Array.isArray(arr) && arr.length >= 2 && Array.isArray(arr[0])) {
    return { I_data: arr[0], Q_data: arr[1] }
  }
  return { I_data: undefined, Q_data: undefined }
}

export function boardSize(count) {
  if (!count) {
    return { width: BOARD_MIN_WIDTH, height: BOARD_MIN_HEIGHT }
  }
  const cols = Math.min(SIGNAL_COLS, count)
  const rows = Math.ceil(count / SIGNAL_COLS)
  return {
    width: Math.max(BOARD_MIN_WIDTH, (cols + 1) * SIGNAL_COL_GAP + 20),
    height: Math.max(BOARD_MIN_HEIGHT, rows * SIGNAL_ROW_GAP + BOARD_PAD),
  }
}

export function layoutNodes(items) {
  const n = items.length
  if (n === 0) return []
  return items.map((item, index) => ({
    ...item,
    x: ((index % SIGNAL_COLS) + 1) * SIGNAL_COL_GAP,
    y: (Math.floor(index / SIGNAL_COLS) + 0.5) * SIGNAL_ROW_GAP,
    radius: SIGNAL_RADIUS,
  }))
}
