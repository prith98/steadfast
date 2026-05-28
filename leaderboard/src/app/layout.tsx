import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Steadfast — AI agent reliability benchmark",
  description:
    "Rigorous, reproducible reliability benchmark for AI agents. v0.1 leaderboard.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
