import { defineConfig } from 'vitepress'

const enNav = [
  {
    text: 'Essentials',
    items: [
      { text: 'Preface', link: '/en/Foreword' },
      { text: 'Tooling', link: '/en/chapter1/wandb_usage_guide' },
      { text: 'Tokenizer', link: '/en/chapter2/chapter2_Tokenizer' },
      { text: 'PyTorch and Resource Accounting', link: '/en/chapter3/chapter3_PyTorch_and_Resource_Accounting' },
      { text: 'Architecture and Training Details', link: '/en/chapter4/chapter4_Architecture_and_Training_Details' },
      { text: 'MOE', link: '/en/chapter5/chapter5_Mixture_of_Experts' },
      { text: 'GPU Optimization', link: '/en/chapter6/chapter6_GPU_and_Optimization' },
      { text: 'Data Engineering', link: '/en/chapter11/chapter11_Data_Engineering' },
      { text: 'Training Pipeline', link: '/en/chapter13/chapter13_Training_Pipeline' },
      { text: 'Evaluation and Benchmarks', link: '/en/chapter12/chapter12_Evaluation_and_Benchmarks' },
    ],
  },
  {
    text: 'Advanced',
    items: [
      { text: 'Distributed Training', link: '/en/chapter8/chapter8_Distributed_Training' },
      { text: 'GPU High Performance Programming', link: '/en/chapter7/chapter7_GPU_High_Performance_Programming' },
      { text: 'Scaling Laws', link: '/en/chapter9/chapter9_Scaling_Laws' },
      { text: 'Inference', link: '/en/chapter10/chapter10_Inference' },
      { text: 'RLVR', link: '/en/chapter14/chapter14_RLVR' },
      { text: 'Multimodal Models', link: '/en/chapter15/chapter15_Multimodal_Models' },
    ],
  },
  {
    text: 'Extra Chapters',
    items: [
      { text: 'Chapter 16 Extended Content', items: [
        { text: '1. What is LLM Reasoning？', link: '/en/chapter16/chapter16_What_is_LLM_Reasoning' },
        { text: '2. LLM Future - LeCun', link: '/en/chapter16/chapter16_LLM_Future_LeCun' },
      ]},
    ]
  },
]

const enSidebar = {
  '/en/': [
    {
      text: 'Chapters',
      items: [
        { text: 'Preface', link: '/en/Foreword' },
        { text: 'Chapter 1 Tooling', link: '/en/chapter1/wandb_usage_guide' },
        { text: 'Chapter 2 Tokenizer', link: '/en/chapter2/chapter2_Tokenizer' },
        { text: 'Chapter 3 PyTorch and Resource Accounting', link: '/en/chapter3/chapter3_PyTorch_and_Resource_Accounting' },
        { text: 'Chapter 4 Language Model Architecture and Training Details', link: '/en/chapter4/chapter4_Architecture_and_Training_Details' },
        { text: 'Chapter 5 Mixture of Experts', link: '/en/chapter5/chapter5_Mixture_of_Experts' },
        { text: 'Chapter 6 GPU and Related Optimization', link: '/en/chapter6/chapter6_GPU_and_Optimization' },
        { text: 'Chapter 7 GPU High Performance Programming', link: '/en/chapter7/chapter7_GPU_High_Performance_Programming' },
        { text: 'Chapter 8 Distributed Training', link: '/en/chapter8/chapter8_Distributed_Training' },
        { text: 'Chapter 9 Scaling Laws', link: '/en/chapter9/chapter9_Scaling_Laws' },
        { text: 'Chapter 10 Inference', link: '/en/chapter10/chapter10_Inference' },
        { text: 'Chapter 11 Data Engineering', link: '/en/chapter11/chapter11_Data_Engineering' },
        { text: 'Chapter 12 Evaluation and Benchmarks', link: '/en/chapter12/chapter12_Evaluation_and_Benchmarks' },
        { text: 'Chapter 13 Basic Training Pipeline for LLMs', link: '/en/chapter13/chapter13_Training_Pipeline' },
        { text: 'Chapter 14 Reinforcement Learning with Verifiable Rewards', link: '/en/chapter14/chapter14_RLVR' },
        { text: 'Chapter 15 Multimodal Models', link: '/en/chapter15/chapter15_Multimodal_Models' },
        { text: 'Chapter 16 Extended Content', items: [
          { text: '1. What is LLM Reasoning?', link: '/en/chapter16/chapter16_What_is_LLM_Reasoning' },
          { text: '2. LLM Future - LeCun', link: '/en/chapter16/chapter16_LLM_Future_LeCun' },
        ]},
      ],
    },
  ],
}

const zhNav = [
  {
    text: '基础必学',
    items: [
      { text: '前言', link: '/前言' },
      { text: '工具使用', link: '/chapter1/wandb使用介绍' },
      { text: '分词器', link: '/chapter2/chapter2_分词器' },
      { text: 'PyTorch与资源核算', link: '/chapter3/chapter3_pytorch与资源核算' },
      { text: '架构与细节', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
      { text: 'MOE', link: '/chapter5/chapter5_混合专家模型' },
      { text: 'GPU优化', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
      { text: '数据工程', link: '/chapter11/chapter11_数据工程' },
      { text: '训练流程', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
      { text: '评估与基准测试', link: '/chapter12/chapter12_评估与基准测试' },
    ],
  },
  {
    text: '进阶选修',
    items: [
      { text: '分布式训练', link: '/chapter8/chapter8_第八章分布式训练' },
      { text: 'GPU高性能编程', link: '/chapter7/chapter7_第七章GPU高性能编程' },
      { text: 'Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
      { text: '推理', link: '/chapter10/chapter10_推理' },
      { text: 'RLVR', link: '/chapter14/chapter14_可验证奖励的强化学习' },
      { text: '多模态模型', link: '/chapter15/chapter15_多模态模型' },
    ],
  },
  {
    text: '扩展内容',
    items: [
      { text: '第16章 扩展内容', items: [
        { text: '1. 什么是LLM推理？', link: '/chapter16/chapter16_什么是LLM推理' },
        { text: '2. LLM 的未来 - Lecun', link: '/chapter16/chapter16_LLM的未来LeCun' },
      ]},
    ],
  },
]

const zhSidebar = {
  '/': [
    {
      text: '章节',
      items: [
        { text: '前言', link: '/前言' },
        { text: '第1章 工具使用', link: '/chapter1/wandb使用介绍' },
        { text: '第2章 分词器', link: '/chapter2/chapter2_分词器' },
        { text: '第3章 PyTorch 与资源核算', link: '/chapter3/chapter3_pytorch与资源核算' },
        { text: '第4章 语言模型架构与训练细节', link: '/chapter4/chapter4_第四章语言模型架构和训练的技术细节' },
        { text: '第5章 混合专家模型', link: '/chapter5/chapter5_混合专家模型' },
        { text: '第6章 GPU 与相关优化', link: '/chapter6/chapter6_第六章GPU和GPU相关的优化' },
        { text: '第7章 GPU 高性能编程', link: '/chapter7/chapter7_第七章GPU高性能编程' },
        { text: '第8章 分布式训练', link: '/chapter8/chapter8_第八章分布式训练' },
        { text: '第9章 Scaling Laws', link: '/chapter9/chapter9_Scaling_Laws' },
        { text: '第10章 推理', link: '/chapter10/推理' },
        { text: '第11章 数据工程', link: '/chapter11/chapter11_数据工程' },
        { text: '第12章 评估与基准测试', link: '/chapter12/chapter12_评估与基准测试' },
        { text: '第13章 大模型的基本训练流程', link: '/chapter13/chapter13_第十三章大模型的基本训练流程' },
        { text: '第14章 可验证奖励的强化学习', link: '/chapter14/chapter14_可验证奖励的强化学习' },
        { text: '第15章 多模态模型', link: '/chapter15/chapter15_多模态模型' },
        { text: '第16章 扩展内容', items: [
          { text: '1. 什么是LLM推理？', link: '/chapter16/chapter16_什么是LLM推理' },
          { text: '2. LLM 的未来 - Lecun', link: '/chapter16/chapter16_LLM的未来LeCun' },
        ]},
      ],
    },
  ],
}

export default defineConfig({
  title: 'Diy-LLM',
  description: '面向中文学习者的大语言模型系统化学习课程。',
  base: '/diy-llm/',
  ignoreDeadLinks: true,
  rewrites: {
    'zh/:rest*': ':rest*',
  },
  head: [
    ['link', { rel: 'icon', href: '/diy-llm/datawhale.png' }],
    ['meta', { name: 'theme-color', content: '#2563eb' }],
    [
      'style',
      {},
      `@page { size: A4; margin: 0; } @media print { html, body { margin: 0 !important; } .VPNav, .VPSidebar, .VPLocalNav, .VPDocAside, .VPDocFooter, .VPFooter, .VPNavScreen { display: none !important; } .VPContent { padding-top: 0 !important; } .VPDoc { padding: 16mm 14mm !important; } .VPDoc .content-container { max-width: none !important; } .vp-doc table { display: table !important; width: 100% !important; table-layout: fixed !important; white-space: normal !important; } .vp-doc th, .vp-doc td { white-space: normal !important; word-break: break-word !important; overflow-wrap: anywhere !important; } .vp-doc pre, .vp-doc code { white-space: pre-wrap !important; word-break: break-all !important; } .vp-doc img { max-width: 100% !important; } }`,
    ],
  ],
  markdown: {
    math: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },
  themeConfig: {
    logo: '/datawhale.png',
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/diy-llm' },
    ],
    outline: {
      level: [2, 3],
    },
  },
  locales: {
    root: {
      label: '中文',
      lang: 'zh-CN',
      srcDir: 'zh',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outlineTitle: '本页目录',
        returnToTopLabel: '返回顶部',
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
      },
    },
    en: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      srcDir: 'en',
      themeConfig: {
        nav: enNav,
        sidebar: enSidebar,
        outlineTitle: 'On this page',
        returnToTopLabel: 'Back to top',
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        docFooter: {
          prev: 'Previous page',
          next: 'Next page',
        },
      },
    },
  },
})
