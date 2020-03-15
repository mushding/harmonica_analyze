import React, { Component } from 'react';
import '../../assets/css/main.css'
import HomePageText from '../../components/HomePageText/HomePageText'
import Footer from '../../components/Footer/Footer'

export default class HomePage extends Component{
    constructor(props){
        super(props);
    }
    componentDidMount(){
    }
    render(){
        return (
            <div>
                {/* <ShowWave/> */}
                <div id="wrapper">
                	<header id="header">
						<div class="logo">
							<span class="icon fa-chart-bar"></span>
						</div>
						<div class="content">
							<div class="inner">
								<HomePageText wait={1000}/>
							</div>
						</div>
						<nav>
							<ul>
								<li><a href="/start">Start</a></li>
							</ul>
						</nav>
					</header>	
					<Footer/>
                </div>
            </div>
        );
    }
}