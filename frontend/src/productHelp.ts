import i18n from 'i18next';

export const helpSteps = ['upload', 'wait', 'select', 'ask'] as const;
export const helpFacts = [
  'custom',
  'training',
  'chunking',
  'privacy',
  'demo',
] as const;

/** Match standalone app-help requests, not questions about document contents. */
export function isProductHelpQuestion(question: string): boolean {
  const normalized = question
    .toLowerCase()
    .trim()
    .replace(/[\s？?！!。.,，：:·]/g, '');
  return (
    /^(?:请问|请介绍|介绍一下)?(?:知源(?:是什么|是干什么的|怎么用|怎么使用|使用方法|使用说明|支持什么|支持哪些文件|支持哪些格式|能做什么|有哪些限制)|什么是知源|怎么使用知源|如何使用知源|你是谁|你能做什么|你支持哪些文件|你支持哪些格式|知源需要重新训练吗|知源只能用于ai开发吗|知源的数据会发送给谁|使用说明|使用指南|帮助|功能说明)$/.test(
      normalized,
    ) ||
    /^(?:whatiszhiyuan|whatis知源|how(?:do|can)iuse(?:zhiyuan|知源)|howtouse(?:zhiyuan|知源)|(?:zhiyuan|知源)(?:help|features|limitations)|userguide)$/.test(
      normalized,
    )
  );
}

/** The same reviewed text powers the help page and local chat responses. */
export function productHelpAnswer(): string {
  return [
    `## ${i18n.t('productHelp.title')}`,
    i18n.t('productHelp.about'),
    `### ${i18n.t('productHelp.stepsTitle')}`,
    ...helpSteps.map(
      (step, index) =>
        `${index + 1}. **${i18n.t(`productHelp.steps.${step}.title`)}**：${i18n.t(`productHelp.steps.${step}.body`)}`,
    ),
    `### ${i18n.t('productHelp.factsTitle')}`,
    ...helpFacts.map(
      (fact) =>
        `- **${i18n.t(`productHelp.facts.${fact}.title`)}**：${i18n.t(`productHelp.facts.${fact}.body`)}`,
    ),
  ].join('\n\n');
}
