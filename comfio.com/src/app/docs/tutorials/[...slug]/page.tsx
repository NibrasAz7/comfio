import { docsNav } from "@/lib/docs";
import { DocPageView, generateDocStaticParams } from "@/components/docs/doc-page-view";

const sectionNav = docsNav.find((s) => s.title === "Tutorials")!;

export function generateStaticParams() {
  return generateDocStaticParams(sectionNav);
}

export default async function TutorialDocPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  return (
    <DocPageView
      slugParts={slug}
      sectionKey="tutorials"
      sectionNav={sectionNav}
      sectionTitle="Tutorials"
      sectionHref="/docs/tutorials"
    />
  );
}
