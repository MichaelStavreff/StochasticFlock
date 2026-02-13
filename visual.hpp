#pragma once
#include <Eigen/Dense>
#include <SFML/Graphics.hpp>
#include <cmath>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "simulation.hpp"

template <int N> class Simulation2d;
template <int N> class Simulation1d;
// struct Simulation2d;

sf::Font loadFont(std::string filename)
{
    sf::Font font;
    if (!font.loadFromFile(filename))
    {
        throw std::invalid_argument("Failed to load font: " + filename);
    }
    return font;
}
template <int N> void init_simulation(Simulation1d<N> &simulation)
{
    int iterations{};
    const int n = simulation.states.rows();
    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(simulation.p.kWIDTH * 0.90, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock");
    title.setPosition(0, simulation.p.kHEIGHT * 0.97);

    sf::VertexArray plot_birds(sf::Quads, n * 4);

    sf::RenderWindow window(sf::VideoMode(simulation.p.kWIDTH, simulation.p.kHEIGHT), "StochasticFlock");
    window.setFramerateLimit(kFRAMERATE);

    for (int i{0}; i < n; ++i)
    {
        plot_birds[i].position =
            sf::Vector2f(simulation.p.kWIDTH / 2 + simulation.states(i, 0), simulation.y_states(i));
    }

    sf::VertexArray axis_line(sf::Lines, 2);
    sf::VertexArray lower_bound(sf::Lines, 2);
    sf::VertexArray upper_bound(sf::Lines, 2);

    axis_line[0].position = sf::Vector2f(0.f, simulation.p.kHEIGHT / 2);
    axis_line[1].position = sf::Vector2f(simulation.p.kWIDTH, simulation.p.kHEIGHT / 2);
    lower_bound[0].position = sf::Vector2f(0.f, simulation.p.kHEIGHT * (0.5 + simulation.p.kBOUND_OFFSET));
    lower_bound[1].position =
        sf::Vector2f(simulation.p.kWIDTH, simulation.p.kHEIGHT * (0.5 + simulation.p.kBOUND_OFFSET));
    upper_bound[0].position = sf::Vector2f(0.f, simulation.p.kHEIGHT * (0.5 - simulation.p.kBOUND_OFFSET));
    upper_bound[1].position =
        sf::Vector2f(simulation.p.kWIDTH, simulation.p.kHEIGHT * (0.5 - simulation.p.kBOUND_OFFSET));
    lower_bound[0].color = sf::Color::Red;
    lower_bound[1].color = sf::Color::Red;
    upper_bound[0].color = sf::Color::Red;
    upper_bound[1].color = sf::Color::Red;

    while (window.isOpen() && iterations < simulation.p.kROUNDS)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // if (simulation.dimensions==1) {
        simulation.update_state();

        for (int i = 0; i < n; ++i)
        {
            int v = i * 4;
            float x_pos = static_cast<float>(simulation.p.kWIDTH / 2 + simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.y_states(i));
            float size = 1.0f;
            sf::Color birdColor = sf::Color::White;
            if (simulation.buffer(i, 2) > 0.5) // Leader
            {
                size = 4.0f;
                birdColor = sf::Color::Yellow;
            }

            plot_birds[v + 0].position = sf::Vector2f(x_pos, y_pos);
            plot_birds[v + 1].position = sf::Vector2f(x_pos + size, y_pos);
            plot_birds[v + 2].position = sf::Vector2f(x_pos + size, y_pos + size);
            plot_birds[v + 3].position = sf::Vector2f(x_pos, y_pos + size);

            for (int j = 0; j < 4; ++j)
            {
                plot_birds[v + j].color = birdColor;
            }
        }
        simulation.shift_back();

        window.clear(sf::Color::Black);

        if (iterations % 100 == 0)
            stats.setString(std::format("n={}\nIter: {}", n, iterations));

        window.draw(stats);
        window.draw(title);
        window.draw(plot_birds);
        window.draw(axis_line);
        window.draw(upper_bound);
        window.draw(lower_bound);
        window.display();
        ++iterations;
    }
}
template <int N> void init_simulation(Simulation2d<N> &simulation)
{
    int iterations{};
    const int n = simulation.states.rows();
    sf::Font font{loadFont("fonts/verdana.ttf")};
    sf::Text stats;
    sf::Text title;
    stats.setFont(font);
    stats.setCharacterSize(15);
    stats.setPosition(simulation.p.kWIDTH_2D * 0.80, 0);
    title.setFont(font);
    title.setCharacterSize(10);
    title.setString("StochasticFlock 2D");
    title.setPosition(0, simulation.p.kHEIGHT_2D * 0.98);
    title.setCharacterSize(15);

    sf::VertexArray plot_birds(sf::Quads, n * 4);

    sf::RenderWindow window(sf::VideoMode(simulation.p.kWIDTH_2D, simulation.p.kHEIGHT_2D), "StochasticFlock 2D");
    window.setFramerateLimit(simulation.p.kFRAMERATE);

    // Axis and box setup
    sf::RectangleShape x_axis_rect(sf::Vector2f(2000000.f, 1.f));
    x_axis_rect.setOrigin(1000000.f, 0.5f);
    x_axis_rect.setPosition(0, 0);
    x_axis_rect.setFillColor(sf::Color::White);

    sf::RectangleShape y_axis_rect(sf::Vector2f(1.f, 2000000.f));
    y_axis_rect.setOrigin(0.5f, 1000000.f);
    y_axis_rect.setPosition(0, 0);
    y_axis_rect.setFillColor(sf::Color::White);

    sf::RectangleShape square(sf::Vector2f(simulation.p.kBOX_SIZE, simulation.p.kBOX_SIZE));
    square.setFillColor(sf::Color::Black); // Changed to transparent to see birds inside
    square.setOutlineThickness(1.f);
    square.setOutlineColor(sf::Color(92, 179, 219));
    square.setOrigin(simulation.p.kBOX_SIZE * 0.5f, simulation.p.kBOX_SIZE * 0.5f);
    square.setPosition(0.f, 0.f);

    sf::Vector2f current_center;
    Eigen::Vector2f target_center;
    sf::View flock_view(sf::FloatRect(0, 0, simulation.p.kWIDTH_2D, simulation.p.kHEIGHT_2D));
    flock_view.setCenter(0, 0);

    float avg_x;
    float avg_y;
    sf::Color color;

    sf::Clock fps_clock;
    float current_fps = 0;

    while (window.isOpen() && iterations < simulation.p.kROUNDS)
    {
        sf::Time dt = fps_clock.restart();
        current_fps = (dt.asSeconds() > 0) ? (1.0f / dt.asSeconds()) : 0;

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        simulation.update_state();

        for (int i = 0; i < n; ++i)
        {
            int v = i * 4;
            float x_pos = static_cast<float>(simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.states(i, 1));

            float size = 2.0f;

            if (simulation.states(i, 4) > 0.5)
            {

                size = 5.0f;
                color = sf::Color::Yellow;
            }
            else
            {
                color = sf::Color::White;
                size = 2.0f;
            }

            bool is_leader = simulation.states(i, 4) > 0.5;

            plot_birds[v + 0].position = sf::Vector2f(x_pos, y_pos);
            plot_birds[v + 1].position = sf::Vector2f(x_pos + size, y_pos);
            plot_birds[v + 2].position = sf::Vector2f(x_pos + size, y_pos + size);
            plot_birds[v + 3].position = sf::Vector2f(x_pos, y_pos + size);

            for (int j = 0; j < 4; ++j)
                plot_birds[v + j].color = color;
        }

        avg_x = simulation.states.col(0).segment(0, 20).mean();
        avg_y = simulation.states.col(1).segment(0, 20).mean();

        simulation.shift_back();
        if (iterations % 25 == 0)
            stats.setString(std::format("FPS: {:.0f}\nn={}\nIter: {}\nBarycenter: ({:.1f},{:.1f})", current_fps, n,
                                        iterations, avg_x, avg_y));
        target_center << avg_x, avg_y;

        // Optional: Smoothly follow (Interpolation)
        current_center = flock_view.getCenter();
        flock_view.setCenter(current_center.x + (target_center.x() - current_center.x) * 0.1f,
                             current_center.y + (target_center.y() - current_center.y) * 0.1f);

        window.clear(sf::Color::Black);

        window.setView(flock_view);
        window.draw(x_axis_rect);
        window.draw(y_axis_rect);
        window.draw(square); // The box stays in the world
        window.draw(plot_birds);

        window.setView(window.getDefaultView());
        window.draw(stats);
        window.draw(title);

        window.display();

        ++iterations;
    }
    std::cout << std::format("\nSimulation ended at iteration {}\nBarycenter: ({},{})\n", iterations, avg_x, avg_y);
}

template <int N> void init_simulation_fontless(Simulation2d<N> &simulation)
{
    int iterations{};
    const int n = simulation.states.rows();

    sf::VertexArray plot_birds(sf::Quads, n * 4);

    sf::RenderWindow window(sf::VideoMode(simulation.p.kWIDTH_2D, simulation.p.kHEIGHT_2D), "StochasticFlock 2D");
    window.setFramerateLimit(simulation.p.kFRAMERATE);

    // Axis and box setup
    sf::RectangleShape x_axis_rect(sf::Vector2f(2000000.f, 1.f));
    x_axis_rect.setOrigin(1000000.f, 0.5f);
    x_axis_rect.setPosition(0, 0);
    x_axis_rect.setFillColor(sf::Color::White);

    sf::RectangleShape y_axis_rect(sf::Vector2f(1.f, 2000000.f));
    y_axis_rect.setOrigin(0.5f, 1000000.f);
    y_axis_rect.setPosition(0, 0);
    y_axis_rect.setFillColor(sf::Color::White);

    sf::RectangleShape square(sf::Vector2f(simulation.p.kBOX_SIZE, simulation.p.kBOX_SIZE));
    square.setFillColor(sf::Color::Black); // Changed to transparent to see birds inside
    square.setOutlineThickness(1.f);
    square.setOutlineColor(sf::Color(92, 179, 219));
    square.setOrigin(simulation.p.kBOX_SIZE * 0.5f, simulation.p.kBOX_SIZE * 0.5f);
    square.setPosition(0.f, 0.f);

    sf::Vector2f current_center;
    Eigen::Vector2f target_center;
    sf::View flock_view(sf::FloatRect(0, 0, simulation.p.kWIDTH_2D, simulation.p.kHEIGHT_2D));
    flock_view.setCenter(0, 0);

    float avg_x;
    float avg_y;
    sf::Color color;

    sf::Clock fps_clock;
    float current_fps = 0;

    while (window.isOpen() && iterations < simulation.p.kROUNDS)
    {
        sf::Time dt = fps_clock.restart();
        current_fps = (dt.asSeconds() > 0) ? (1.0f / dt.asSeconds()) : 0;

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        simulation.update_state();

        for (int i = 0; i < n; ++i)
        {
            int v = i * 4;
            float x_pos = static_cast<float>(simulation.states(i, 0));
            float y_pos = static_cast<float>(simulation.states(i, 1));

            float size = 2.0f;

            if (simulation.states(i, 4) > 0.5)
            {

                size = 5.0f;
                color = sf::Color::Yellow;
            }
            else
            {
                color = sf::Color::White;
                size = 2.0f;
            }

            bool is_leader = simulation.states(i, 4) > 0.5;

            plot_birds[v + 0].position = sf::Vector2f(x_pos, y_pos);
            plot_birds[v + 1].position = sf::Vector2f(x_pos + size, y_pos);
            plot_birds[v + 2].position = sf::Vector2f(x_pos + size, y_pos + size);
            plot_birds[v + 3].position = sf::Vector2f(x_pos, y_pos + size);

            for (int j = 0; j < 4; ++j)
                plot_birds[v + j].color = color;
        }

        avg_x = simulation.states.col(0).segment(0, 20).mean();
        avg_y = simulation.states.col(1).segment(0, 20).mean();

        simulation.shift_back();
        target_center << avg_x, avg_y;

        // Optional: Smoothly follow (Interpolation)
        current_center = flock_view.getCenter();
        flock_view.setCenter(current_center.x + (target_center.x() - current_center.x) * 0.1f,
                             current_center.y + (target_center.y() - current_center.y) * 0.1f);

        window.clear(sf::Color::Black);

        window.setView(flock_view);
        window.draw(x_axis_rect);
        window.draw(y_axis_rect);
        window.draw(square); // The box stays in the world
        window.draw(plot_birds);

        window.setView(window.getDefaultView());

        window.display();

        ++iterations;
    }
    std::cout << std::format("\nSimulation ended at iteration {}\nBarycenter: ({},{})\n", iterations, avg_x, avg_y);
}