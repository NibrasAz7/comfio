declare module "react-katex" {
  import * as React from "react";

  interface KaTeXProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
    settings?: Record<string, unknown>;
  }

  export class InlineMath extends React.Component<KaTeXProps> {}
  export class BlockMath extends React.Component<KaTeXProps> {}
}
