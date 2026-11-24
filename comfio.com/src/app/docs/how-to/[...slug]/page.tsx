import { docsNav } from "@/lib/docs";
import { DocPageView, generateDocStaticParams } from "@/components/docs/doc-page-view";

const sectionNav = docsNav.find((s) => s.title === "How-To Guides")!;

export function generateStaticParams() {
  return generateDocStaticParams(sectionNav);
}

export default async function HowToDocPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  return (
    <DocPageView
      slugParts={slug}
      sectionKey="how_to"
      sectionNav={sectionNav}
      sectionTitle="How-To Guides"
      sectionHref="/docs/how-to"
    />
  );
}
