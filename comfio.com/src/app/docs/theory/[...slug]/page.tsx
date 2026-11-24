import { docsNav } from "@/lib/docs";
import { DocPageView, generateDocStaticParams } from "@/components/docs/doc-page-view";

const sectionNav = docsNav.find((s) => s.title === "Theory")!;

export function generateStaticParams() {
  return generateDocStaticParams(sectionNav);
}

export default async function TheoryDocPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  return (
    <DocPageView
      slugParts={slug}
      sectionKey="theory"
      sectionNav={sectionNav}
      sectionTitle="Theory"
      sectionHref="/docs/theory"
    />
  );
}
