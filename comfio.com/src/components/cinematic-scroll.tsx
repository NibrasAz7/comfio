"use client";

import { useEffect, useRef } from "react";

export function CinematicScroll() {
  const lenisRef = useRef<any>(null);
  const gsapRef = useRef<any>(null);

  useEffect(() => {
    let mounted = true;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    async function setup() {
      if (reduced) return;

      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import("gsap"),
        import("gsap/ScrollTrigger"),
      ]);
      const Lenis = (await import("lenis")).default;

      if (!mounted) return;

      gsap.registerPlugin(ScrollTrigger);
      gsapRef.current = gsap;

      // Lenis smooth scroll driven by GSAP ticker
      const lenis = new Lenis({
        duration: 1.2,
        easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: true,
      });
      lenisRef.current = lenis;

      lenis.on("scroll", ScrollTrigger.update);
      gsap.ticker.add((time: number) => lenis.raf(time * 1000));
      gsap.ticker.lagSmoothing(0);

      // Expose for debugging
      (window as any).lenis = lenis;

      // Scroll progress bar
      const progressEl = document.getElementById("scroll-progress");
      if (progressEl) {
        ScrollTrigger.create({
          start: 0,
          end: "max",
          onUpdate: (self) => {
            progressEl.style.transform = `scaleX(${self.progress})`;
          },
        });
      }

      // Header hide/show on scroll direction
      const header = document.getElementById("site-header");
      if (header) {
        ScrollTrigger.create({
          start: 0,
          end: "max",
          onUpdate: (self) => {
            if (self.scroll() > 140) {
              header.classList.toggle("header-hidden", self.direction === 1);
            } else {
              header.classList.remove("header-hidden");
            }
            header.classList.toggle("header-solid", self.scroll() > 40);
          },
        });
      }

      // Hero entrance animations
      const heroTl = gsap.timeline({ defaults: { ease: "power3.out" } });
      heroTl
        .from("[data-hero-badge]", { y: 20, opacity: 0, duration: 0.6 })
        .from("[data-hero-title] > *", { y: 40, opacity: 0, duration: 0.8, stagger: 0.15 }, "-=0.3")
        .from("[data-hero-desc]", { y: 20, opacity: 0, duration: 0.6 }, "-=0.4")
        .from("[data-hero-actions]", { y: 20, opacity: 0, duration: 0.5 }, "-=0.3")
        .from("[data-hero-domains] > *", { y: 15, opacity: 0, duration: 0.4, stagger: 0.08 }, "-=0.2");

      // Section reveal animations
      gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((el) => {
        gsap.from(el, {
          y: 60,
          opacity: 0,
          duration: 0.8,
          ease: "power2.out",
          scrollTrigger: {
            trigger: el,
            start: "top 85%",
            end: "top 50%",
            scrub: false,
            toggleActions: "play none none reverse",
          },
        });
      });

      // Feature cards stagger
      gsap.utils.toArray<HTMLElement>("[data-feature-card]").forEach((el, i) => {
        gsap.from(el, {
          y: 40,
          opacity: 0,
          duration: 0.5,
          delay: i * 0.05,
          scrollTrigger: {
            trigger: el,
            start: "top 90%",
            toggleActions: "play none none reverse",
          },
        });
      });

      // Ambient background color shifts
      const ambient = document.getElementById("ambient");
      if (ambient) {
        const sections = [
          { trigger: "#hero", color: "rgba(250,245,238,0)" },
          { trigger: "#quickstart", color: "rgba(240,235,225,0.3)" },
          { trigger: "#features", color: "rgba(235,230,220,0.4)" },
          { trigger: "#standards", color: "rgba(240,235,225,0.2)" },
          { trigger: "#docs-cta", color: "rgba(235,230,220,0.3)" },
          { trigger: "#final-cta", color: "rgba(245,240,232,0.1)" },
        ];
        sections.forEach((s) => {
          ScrollTrigger.create({
            trigger: s.trigger,
            start: "top 60%",
            end: "bottom 40%",
            onEnter: () => gsap.to(ambient, { backgroundColor: s.color, duration: 1.1 }),
            onEnterBack: () => gsap.to(ambient, { backgroundColor: s.color, duration: 1.1 }),
          });
        });
      }

      // Glow opacity shifts
      const glow = document.getElementById("glow");
      if (glow) {
        ScrollTrigger.create({
          trigger: "#hero",
          start: "top top",
          end: "bottom top",
          scrub: true,
          onUpdate: (self) => {
            glow.style.opacity = String(0.6 - self.progress * 0.4);
          },
        });
      }
    }

    setup();

    return () => {
      mounted = false;
      if (lenisRef.current) lenisRef.current.destroy();
      if (gsapRef.current) gsapRef.current.kill();
    };
  }, []);

  return null;
}
