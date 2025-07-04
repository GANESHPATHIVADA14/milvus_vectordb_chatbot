"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function ChatPage() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(
        `http://localhost:8000/chat?query=${encodeURIComponent(query)}`
      );
      const data = await res.json();
      setResponse(data.response || data.error || "No response");
    } catch (error) {
      setResponse("Error contacting backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
      <Card className="w-full max-w-xl shadow-md">
        <CardContent className="space-y-4 p-6">
          <h1 className="text-2xl font-bold">🧠 Gemini Chatbot</h1>

          <Input
            placeholder="Ask something about attention..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />

          <Button onClick={handleSubmit} disabled={loading} className="w-full">
            {loading ? "Thinking..." : "Ask"}
          </Button>

          {response && (
            <div className="bg-muted p-4 rounded text-sm whitespace-pre-wrap">
              <strong>Bot:</strong> {response}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
