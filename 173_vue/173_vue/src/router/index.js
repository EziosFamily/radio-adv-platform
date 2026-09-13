import { createRouter, createWebHistory } from 'vue-router'
import home from '../components/home.vue'
import tab1 from '../components/tab1.vue'
import tab2 from '../components/tab2.vue'
import tab3 from '../components/tab3.vue'
import tab4 from '../components/tab4.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: home
    },
    {
      path: '/tab1',
      name: 'tab1',
      component: tab1
    },
    {
      path: '/tab2',
      name: 'tab2',
      component: tab2
    },
    {
      path: '/tab3',
      name: 'tab3',
      component: tab3
    },
    {
      path: '/tab4',
      name: 'tab4',
      component: tab4
    },
  ]
})

export default router
