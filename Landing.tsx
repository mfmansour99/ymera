import { Link } from 'wouter';
import { Zap, Activity, Users, Shield, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Landing() {
  const features = [
    {
      icon: Activity,
      title: 'AI Agent Orchestration',
      description: 'Seamlessly coordinate multiple AI models and agents across OpenAI, Anthropic, and Google.',
    },
    {
      icon: Users,
      title: 'Workflow Management',
      description: 'Design, deploy, and monitor complex AI workflows with visual builders and real-time monitoring.',
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Role-based access control, audit logging, and comprehensive security monitoring.',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-secondary/20">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-primary">Ymera</span>
            </div>
            <Button
              onClick={() => window.location.href = '/api/login'}
              className="bg-primary hover:bg-primary/90 text-primary-foreground"
              data-testid="login-button"
            >
              Sign In
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="mb-8">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent animate-pulse">
                Ymera AI
              </span>
              <br />
              <span className="text-foreground">Orchestration Platform</span>
            </h1>
            <p className="mt-6 text-xl text-muted-foreground max-w-3xl mx-auto">
              The enterprise-grade platform for orchestrating AI agents, managing workflows, 
              and monitoring AI operations at scale. Built for teams that demand reliability and security.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
            <Button
              size="lg"
              onClick={() => window.location.href = '/api/login'}
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-4 text-lg"
              data-testid="get-started-button"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>
            <Button
              variant="outline"
              size="lg"
              asChild
              className="px-8 py-4 text-lg border-border hover:bg-accent/20"
            >
              <a href="#features">Learn More</a>
            </Button>
          </div>

          {/* Hero Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            <div className="text-center">
              <div className="text-3xl font-bold text-primary">10K+</div>
              <div className="text-muted-foreground">Tasks Executed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-accent">99.9%</div>
              <div className="text-muted-foreground">Uptime</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-primary">24/7</div>
              <div className="text-muted-foreground">Monitoring</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-secondary/10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
              Enterprise AI Orchestration
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need to deploy, manage, and scale AI operations in your organization.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <Card
                key={feature.title}
                className="bg-card/50 backdrop-blur-sm border-border/50 hover:bg-card/70 transition-all duration-300 hover:scale-105"
                data-testid={`feature-card-${index}`}
              >
                <CardHeader>
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle className="text-xl font-semibold text-foreground">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Ready to Transform Your AI Operations?
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join organizations already using Ymera to orchestrate their AI workflows 
            with confidence and reliability.
          </p>
          <Button
            size="lg"
            onClick={() => window.location.href = '/api/login'}
            className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-4 text-lg"
            data-testid="cta-button"
          >
            Start Your Journey
            <ArrowRight className="ml-2 w-5 h-5" />
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 sm:px-6 lg:px-8 border-t border-border/30">
        <div className="max-w-7xl mx-auto text-center text-muted-foreground">
          <p>&copy; 2025 Ymera AI Platform. Enterprise-grade AI orchestration.</p>
        </div>
      </footer>
    </div>
  );
}
