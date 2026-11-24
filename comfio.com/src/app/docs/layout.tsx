import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import { DocsSidebar } from "@/components/docs/docs-sidebar";
import { DocsSearch } from "@/components/docs/docs-search";
import { MobileSidebar } from "@/components/docs/mobile-sidebar";

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen">
      <Header />
      <div className="pt-14 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex gap-8">
        {/* Desktop sidebar */}
        <aside className="hidden lg:block w-64 shrink-0 py-8 sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto">
          <div className="mb-6">
            <DocsSearch />
          </div>
          <DocsSidebar />
        </aside>

        {/* Content */}
        <main className="flex-1 min-w-0 py-8 max-w-4xl">
          <MobileSidebar />
          {children}
        </main>
      </div>
      <Footer />
    </div>
  );
}
