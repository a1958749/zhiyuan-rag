import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { helpFacts, helpSteps } from './productHelp';

export default function UserGuide() {
  const { t } = useTranslation();
  return (
    <main className="text-foreground h-full overflow-y-auto p-6">
      <div className="mx-auto flex max-w-3xl flex-col gap-6 pb-10">
        <div>
          <h1 className="text-2xl font-semibold">{t('productHelp.title')}</h1>
          <p className="text-muted-foreground mt-2 text-sm">
            {t('productHelp.fixedNotice')}
          </p>
          <p className="mt-4 text-sm leading-6">{t('productHelp.about')}</p>
        </div>
        <section aria-labelledby="start-heading">
          <h2 id="start-heading" className="mb-3 text-lg font-semibold">
            {t('productHelp.stepsTitle')}
          </h2>
          <ol className="grid gap-3 sm:grid-cols-2">
            {helpSteps.map((step, index) => (
              <li key={step}>
                <Card variant="subtle" className="h-full">
                  <h3 className="font-semibold">
                    {index + 1}. {t(`productHelp.steps.${step}.title`)}
                  </h3>
                  <p className="leading-6">
                    {t(`productHelp.steps.${step}.body`)}
                  </p>
                </Card>
              </li>
            ))}
          </ol>
          <div className="mt-4 flex flex-wrap gap-3">
            <Button asChild>
              <Link to="/settings/knowledge">
                {t('productHelp.openKnowledge')}
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link to="/c/new">{t('productHelp.openChat')}</Link>
            </Button>
            <Button variant="ghost" asChild>
              <Link to="/settings/general">{t('productHelp.language')}</Link>
            </Button>
          </div>
        </section>
        <section aria-labelledby="example-heading">
          <h2 id="example-heading" className="mb-3 text-lg font-semibold">
            {t('productHelp.exampleTitle')}
          </h2>
          <Card variant="filled">
            <p className="leading-6">{t('productHelp.example')}</p>
            <p className="leading-6">{t('productHelp.exampleQuestion')}</p>
            <Button variant="outline" asChild className="self-start">
              <a href="/examples/zhiyuan-first-document.txt" download>
                {t('productHelp.downloadExample')}
              </a>
            </Button>
          </Card>
        </section>
        <section aria-labelledby="facts-heading">
          <h2 id="facts-heading" className="mb-3 text-lg font-semibold">
            {t('productHelp.factsTitle')}
          </h2>
          <div className="flex flex-col gap-3">
            {helpFacts.map((fact) => (
              <Card key={fact} variant="subtle">
                <h3 className="font-semibold">
                  {t(`productHelp.facts.${fact}.title`)}
                </h3>
                <p className="leading-6">
                  {t(`productHelp.facts.${fact}.body`)}
                </p>
              </Card>
            ))}
          </div>
        </section>
        <section>
          <h2 className="mb-2 text-lg font-semibold">
            {t('productHelp.troubleshootingTitle')}
          </h2>
          <p className="text-sm leading-6">
            {t('productHelp.troubleshooting')}
          </p>
        </section>
      </div>
    </main>
  );
}
