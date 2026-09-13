import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5000'

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 180000,
})

export const DEFAULT_ATTACK_METHODS = [
  'FGSM',
  'PGD',
  'BIM',
  'MI_FGSM',
  'NAM',
  'MPDSM',
  'TSMIFGSM',
]

export const DEFAULT_DETECT_METHODS = [
  'FGSM',
  'PGD',
  'BIM',
  'MI_FGSM',
  'NAM',
  'MPDSM',
  'TSMIFGSM',
]
