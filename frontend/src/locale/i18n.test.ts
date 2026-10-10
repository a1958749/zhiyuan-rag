import { beforeAll, describe, expect, it, vi } from 'vitest';

import type { i18n as I18n } from 'i18next';

// Old builds stored the string "undefined" as the language; detection must
// skip it and use the product's Chinese default.
let i18n: I18n;

beforeAll(async () => {
  localStorage.setItem('docsgpt-locale', 'undefined');
  Object.defineProperty(navigator, 'languages', {
    value: ['ru-RU', 'en-US'],
    configurable: true,
  });
  i18n = (await import('./i18n')).default;
});

describe('i18n language detection', () => {
  it('uses Chinese when no valid preference exists, regardless of browser language', () => {
    expect(i18n.language).toBe('zh');
    expect(localStorage.getItem('docsgpt-locale')).toBe('zh');
  });

  it('restores English after a reload and persists a switch back to Chinese', async () => {
    localStorage.setItem('docsgpt-locale', 'en');
    vi.resetModules();
    const restored = (await import('./i18n')).default;
    expect(restored.language).toBe('en');
    expect(restored.t('newChat')).toBe('New Chat');
    await restored.changeLanguage('zh');
    expect(localStorage.getItem('docsgpt-locale')).toBe('zh');
    vi.resetModules();
    const reloaded = (await import('./i18n')).default;
    expect(reloaded.language).toBe('zh');
    expect(reloaded.t('newChat')).toBe('新聊天');
  });

  it('keeps the app codes that are not BCP 47 tags', async () => {
    await i18n.changeLanguage('zhTW');
    expect(i18n.language).toBe('zhTW');
    await i18n.changeLanguage('jp');
    expect(i18n.language).toBe('jp');
  });
});

describe('html lang', () => {
  it('follows the UI language as a BCP 47 tag', async () => {
    // A page that declares English while showing Spanish invites the
    // browser's translator, whose DOM rewrites crash React.
    await i18n.changeLanguage('de');
    expect(document.documentElement.lang).toBe('de');
    await i18n.changeLanguage('zhTW');
    expect(document.documentElement.lang).toBe('zh-TW');
    await i18n.changeLanguage('jp');
    expect(document.documentElement.lang).toBe('ja');
  });
});
