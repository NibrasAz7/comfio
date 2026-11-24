import { docsNav } from "@/lib/docs";
import { DocPageView, generateDocStaticParams } from "@/components/docs/doc-page-view";

const sectionNav = docsNav.find((s) => s.title === "API Reference")!;

export function generateStaticParams() {
  return generateDocStaticParams(sectionNav);
}

export default async function ReferenceDocPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  return (
    <DocPageView
      slugParts={slug}
      sectionKey="reference"
      sectionNav={sectionNav}
      sectionTitle="API Reference"
      sectionHref="/docs/reference"
    />
  );
}
