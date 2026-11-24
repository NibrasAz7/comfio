import { Header } from "@/components/header";
import { Footer } from "@/components/footer";
import { Playground } from "@/components/playground/playground";

export const metadata = {
  title: "Playground — comfio",
  description: "Interactive IEQ calculator. Adjust thermal, visual, acoustic, and IAQ parameters to see real-time comfort scores.",
};

export default function PlaygroundPage() {
  return (
    <div className="min-h-screen">
      <Header />
      <main className="pt-24 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto mb-10 text-center">
          <h1 className="text-3xl sm:text-4xl font-bold mb-3">IEQ Playground</h1>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Adjust environmental parameters and see how comfio computes domain scores and a unified Global IEQ Index.
            This is a simplified demo — the real library uses validated standards (ISO 7730, ASHRAE 55, EN 12464-1, ASHRAE 62.1).
          </p>
        </div>
        <Playground />
      </main>
      <Footer />
    </div>
  );
}
